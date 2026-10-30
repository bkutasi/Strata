package server

import (
	"context"
	"errors"

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
