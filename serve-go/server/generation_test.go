package server

import (
	"context"
	"errors"
	"testing"

	"strata/engine"
	"strata/outputparser"
)

type pumpEngine struct {
	tokens []int
	err    error
	finish string
}

func (e pumpEngine) Alive() bool       { return true }
func (e pumpEngine) ContextLimit() int { return 4096 }
func (e pumpEngine) GenerateWithOptions(ctx context.Context, _ []int, _ int, opt engine.RequestOptions) (engine.Result, error) {
	for _, id := range e.tokens {
		if err := opt.OnToken(id); err != nil {
			return engine.Result{}, err
		}
	}
	finish := e.finish
	if finish == "" {
		finish = "stop"
	}
	return engine.Result{Finish: finish}, e.err
}

type pumpParser struct {
	finish string
}

func (*pumpParser) Feed(s string) ([]outputparser.Event, error) {
	return []outputparser.Event{{Kind: "content", Text: s}}, nil
}
func (p *pumpParser) Finish(reason string) ([]outputparser.Event, error) {
	p.finish = reason
	return nil, nil
}

func TestGenerateStopTokenAndStopString(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser := &pumpParser{}
	var got string
	out, err := generate(context.Background(), pumpEngine{tokens: []int{'a', 257, 'b'}}, tok, parser, []int{1}, 8, engine.Sampling{}, map[int]bool{257: true}, func(events []outputparser.Event) error {
		for _, event := range events {
			got += event.Text
		}
		return nil
	})
	if err != nil || got != "a" || out.count != 2 || !out.stopped || parser.finish != "stop" {
		t.Fatalf("stop token: out=%+v got=%q finish=%q err=%v", out, got, parser.finish, err)
	}

	parser = &pumpParser{}
	got = ""
	out, err = generate(context.Background(), pumpEngine{tokens: []int{'x', '!', 'y'}}, tok, parser, []int{1}, 8, engine.Sampling{}, nil, func(events []outputparser.Event) error {
		for _, event := range events {
			got += event.Text
			if event.Text == "!" {
				return stopGeneration
			}
		}
		return nil
	})
	if err != nil || got != "x!" || out.count != 2 || !out.stopped || parser.finish != "stop" {
		t.Fatalf("stop string: out=%+v got=%q finish=%q err=%v", out, got, parser.finish, err)
	}
}

func TestGenerateEarlyStopFinishesParserAsStop(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	for _, tc := range []struct {
		name      string
		tokens    []int
		stopIDs   map[int]bool
		stopEvent bool
	}{
		{name: "stop token", tokens: []int{257}, stopIDs: map[int]bool{257: true}},
		{name: "stop string", tokens: []int{'!'}, stopEvent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parser := &pumpParser{}
			_, err := generate(context.Background(), pumpEngine{tokens: tc.tokens, finish: "length"}, tok, parser, []int{1}, 8, engine.Sampling{}, tc.stopIDs, func(events []outputparser.Event) error {
				if tc.stopEvent && len(events) > 0 {
					return stopGeneration
				}
				return nil
			})
			if err != nil || parser.finish != "stop" {
				t.Fatalf("parser finish=%q err=%v; want stop", parser.finish, err)
			}
		})
	}
}

func TestGeneratePropagatesParserAndEngineErrors(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	want := errors.New("callback failed")
	_, err := generate(context.Background(), pumpEngine{tokens: []int{'a'}}, tok, &pumpParser{}, []int{1}, 8, engine.Sampling{}, nil, func([]outputparser.Event) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("callback error = %v, want %v", err, want)
	}

	engineErr := errors.New("engine failed")
	out, err := generate(context.Background(), pumpEngine{tokens: []int{'a'}, err: engineErr}, tok, &pumpParser{}, []int{1}, 8, engine.Sampling{}, nil, func([]outputparser.Event) error { return nil })
	if !errors.Is(err, engineErr) || out.count != 1 {
		t.Fatalf("engine error: out=%+v err=%v", out, err)
	}
}
