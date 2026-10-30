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

// generate pumps engine tokens through the tokenizer and parser. The callback
// owns event routing and may return stopGeneration to drain inference.
func generate(ctx context.Context, e Engine, tokenizer *modeltokenizer.Tokenizer, parser generationParser, ids []int, max int, sampling engine.Sampling, stopIDs map[int]bool, callback func([]outputparser.Event) error) (generationResult, error) {
	var out generationResult
	decoder := streamtext.Decoder{}
	stoppedToken := false
	result, err := e.GenerateWithOptions(ctx, ids, max, engine.RequestOptions{Sampling: sampling, OnToken: func(id int) error {
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
		return err
	}})
	out.result = result
	out.stopped = stoppedToken
	if errors.Is(err, engine.ErrStopGeneration) {
		err = nil
	}
	if err != nil {
		return out, err
	}
	out.cached, err = cachedPromptTokens(result, len(ids), stoppedToken)
	if err != nil {
		return out, err
	}
	out.timings, err = requestTimings(result, len(ids), out.count, out.cached)
	if err != nil {
		return out, err
	}
	finish := result.Finish
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
