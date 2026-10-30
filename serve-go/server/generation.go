package server

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"strata/engine"
	"strata/outputparser"
	"strata/serve-go/modeltokenizer"
	"strata/streamtext"
)

type generationParser interface {
	Feed(string) ([]outputparser.Event, error)
	Finish(string) ([]outputparser.Event, error)
}

type generationResult struct {
	result  engine.Result
	count   int
	stopped bool
	cached  int
	timings map[string]any
}

// DONE extras contain drafts accepted/offered, reused tokens, and optionally
// prompt tokens actually read. Older engines omit those fields.
func cachedPromptTokens(result engine.Result, prompt int, stopped bool) (int, error) {
	cached := 0
	if len(result.Extra) >= 3 {
		n, err := strconv.Atoi(result.Extra[2])
		if err != nil || n < 0 {
			return 0, fmt.Errorf("invalid engine reused token count %q", result.Extra[2])
		}
		cached = n
	}
	if len(result.Extra) >= 9 && !stopped && result.Finish != "cancel" {
		read, err := strconv.Atoi(result.Extra[8])
		if err != nil || read < 0 {
			return 0, fmt.Errorf("invalid engine prompt read count %q", result.Extra[8])
		}
		if prompt-read > cached {
			cached = prompt - read
		}
	}
	if cached > prompt {
		cached = prompt
	}
	return cached, nil
}

const reasoningWrapUp = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"

// generate pumps tokens through the parser and resumes after a thinking budget.
// The callback may return stopGeneration to drain inference.
func generate(ctx context.Context, e Engine, tokenizer *modeltokenizer.Tokenizer, parser generationParser, ids []int, max int, sampling engine.Sampling, stopIDs map[int]bool, budget int, callback func([]outputparser.Event) error) (generationResult, error) {
	var out generationResult
	decoder := streamtext.Decoder{}
	stoppedToken := false
	status, hasStatus := parser.(interface{ Status() outputparser.Status })
	if budget > 0 {
		if !hasStatus {
			return out, errors.New("reasoning budget requires parser status")
		}
		if status.Status().State != "reasoning" {
			budget = 0
		}
	}
	if budget > 0 {
		reserve := min(maxInt(512, max/4), max/2)
		budget = min(budget, maxInt(1, max-reserve))
	}
	prompt := ids
	thought := 0
	first := true
	for {
		var segment []int
		wrap := false
		result, err := e.GenerateWithOptions(ctx, prompt, max-out.count, engine.RequestOptions{Sampling: sampling, OnToken: func(id int) error {
			out.count++
			if out.count > max {
				return errors.New("engine exceeded output token limit")
			}
			if stoppedToken {
				return nil
			}
			if stopIDs[id] {
				stoppedToken = true
				return engine.ErrStopGeneration
			}
			if budget > 0 {
				segment = append(segment, id)
			}
			raw, err := tokenizer.TokenBytes(id)
			if err != nil {
				return err
			}
			events, err := parser.Feed(decoder.Push(raw))
			if err != nil {
				return err
			}
			err = callback(events)
			if errors.Is(err, stopGeneration) {
				stoppedToken = true
				return engine.ErrStopGeneration
			}
			if err != nil {
				return err
			}
			if budget > 0 {
				state := status.Status()
				if state.State == "reasoning" {
					thought++
					if thought >= budget && state.Clean && !decoder.Pending() {
						wrap = true
						return engine.ErrStopGeneration
					}
				}
			}
			return nil
		}})
		if errors.Is(err, engine.ErrStopGeneration) {
			err = nil
		}
		if err != nil {
			return out, err
		}
		if first {
			out.result = result
			out.cached, err = cachedPromptTokens(result, len(ids), stoppedToken || wrap)
			if err != nil {
				return out, err
			}
			first = false
		} else {
			if err = mergeGenerationResult(&out.result, result); err != nil {
				return out, err
			}
		}
		if !wrap {
			break
		}
		out.result.Tokens = nil
		extra, err := tokenizer.Encode(reasoningWrapUp, true, nil)
		if err != nil {
			return out, err
		}
		if max-out.count-len(extra) < 1 {
			out.result.Finish = "length"
			break
		}
		for _, id := range extra {
			out.count++
			raw, err := tokenizer.TokenBytes(id)
			if err != nil {
				return out, err
			}
			events, err := parser.Feed(decoder.Push(raw))
			if err != nil {
				return out, err
			}
			if err = callback(events); errors.Is(err, stopGeneration) {
				stoppedToken = true
				break
			} else if err != nil {
				return out, err
			}
		}
		if stoppedToken {
			break
		}
		continued := make([]int, 0, len(prompt)+len(segment)+len(extra))
		continued = append(continued, prompt...)
		continued = append(continued, segment...)
		prompt = append(continued, extra...)
		budget = 0
	}
	out.stopped = stoppedToken
	var err error
	out.timings, err = requestTimings(out.result, len(ids), out.count, out.cached)
	if err != nil {
		return out, err
	}
	finish := out.result.Finish
	if stoppedToken {
		finish = "stop"
	}
	events, err := parser.Finish(finish)
	if err != nil {
		return out, err
	}
	if err = callback(events); errors.Is(err, stopGeneration) {
		err = nil
		out.result.Finish = "stop"
	}
	if err != nil {
		return out, err
	}
	return out, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mergeGenerationResult(first *engine.Result, next engine.Result) error {
	first.Finish = next.Finish
	first.Generated += next.Generated
	first.PromptMS += next.PromptMS
	first.DecodeMS += next.DecodeMS
	if len(first.Extra) < 3 && len(next.Extra) >= 3 {
		first.Extra = []string{"0", "0", "0"}
	}
	if len(first.Extra) >= 3 && len(next.Extra) >= 3 {
		for i := 0; i < 2; i++ {
			a, errA := strconv.Atoi(first.Extra[i])
			b, errB := strconv.Atoi(next.Extra[i])
			if errA != nil || errB != nil || a < 0 || b < 0 || b > int(^uint(0)>>1)-a {
				return errors.New("invalid engine draft count")
			}
			first.Extra[i] = strconv.Itoa(a + b)
		}
	}
	return nil
}
