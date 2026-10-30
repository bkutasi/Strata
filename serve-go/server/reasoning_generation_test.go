package server

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"strata/engine"
	"strata/outputparser"
)

type budgetEngine struct {
	segments [][]int
	prompts  [][]int
	limits   []int
}

func (*budgetEngine) Alive() bool       { return true }
func (*budgetEngine) ContextLimit() int { return 4096 }
func (e *budgetEngine) GenerateWithOptions(ctx context.Context, ids []int, limit int, opts engine.RequestOptions) (engine.Result, error) {
	e.prompts = append(e.prompts, slices.Clone(ids))
	e.limits = append(e.limits, limit)
	if len(e.prompts) > len(e.segments) {
		return engine.Result{}, errors.New("unexpected extra generation")
	}
	tokens := e.segments[len(e.prompts)-1]
	for i, id := range tokens {
		if err := ctx.Err(); err != nil {
			return engine.Result{}, err
		}
		if err := opts.OnToken(id); err != nil {
			if errors.Is(err, engine.ErrStopGeneration) {
				return engine.Result{Finish: "stop", Generated: i + 1}, nil
			}
			return engine.Result{}, err
		}
	}
	return engine.Result{Finish: "stop", Generated: len(tokens)}, nil
}

func TestReasoningBudgetWaitsForCleanBoundary(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	for _, tc := range []struct {
		name   string
		tokens []int
	}{
		{"split tag", []int{'<', 'x', 'a'}},
		{"split UTF-8", []int{0xc3, 0xa9, 'a'}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parser, err := outputparser.New(outputparser.Options{Thinking: true})
			if err != nil {
				t.Fatal(err)
			}
			e := &budgetEngine{segments: [][]int{tc.tokens, {'z'}}}
			_, err = generate(context.Background(), e, tok, parser, []int{'H'}, 200, engine.Sampling{}, nil, 1,
				func([]outputparser.Event) error { return nil })
			if err != nil || len(e.prompts) != 2 || len(e.prompts[1]) < 3 ||
				!slices.Equal(e.prompts[1][:3], []int{'H', tc.tokens[0], tc.tokens[1]}) {
				t.Fatalf("prompts=%v err=%v", e.prompts, err)
			}
		})
	}
}

func TestReasoningBudgetDoesNotResumeAfterThinkingEnds(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true})
	if err != nil {
		t.Fatal(err)
	}
	var tokens []int
	for _, b := range []byte("abc</think>done") {
		tokens = append(tokens, int(b))
	}
	e := &budgetEngine{segments: [][]int{tokens}}
	var content strings.Builder
	out, err := generate(context.Background(), e, tok, parser, []int{'H'}, 200, engine.Sampling{}, nil, 100,
		func(events []outputparser.Event) error {
			for _, event := range events {
				if event.Kind == "content" {
					content.WriteString(event.Text)
				}
			}
			return nil
		})
	if err != nil || len(e.prompts) != 1 || content.String() != "done" || out.result.Finish != "stop" {
		t.Fatalf("out=%+v prompts=%v content=%q err=%v", out, e.prompts, content.String(), err)
	}
}

func TestReasoningBudgetRequiresParserStatus(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	_, err := generate(context.Background(), pumpEngine{tokens: []int{'a'}}, tok, &pumpParser{},
		[]int{'H'}, 200, engine.Sampling{}, nil, 3, func([]outputparser.Event) error { return nil })
	if err == nil {
		t.Fatal("accepted a reasoning budget without parser status")
	}
}

func TestReasoningBudgetAggregatesEngineTimings(t *testing.T) {
	first := engine.Result{Generated: 3, PromptMS: 4, DecodeMS: 5, Finish: "stop", Extra: []string{"2", "4", "9"}}
	second := engine.Result{Generated: 7, PromptMS: 6, DecodeMS: 8, Finish: "length", Extra: []string{"3", "5", "20"}}
	if err := mergeGenerationResult(&first, second); err != nil || first.Generated != 10 || first.PromptMS != 10 ||
		first.DecodeMS != 13 || first.Finish != "length" || !slices.Equal(first.Extra, []string{"5", "9", "9"}) {
		t.Fatalf("merged=%+v err=%v", first, err)
	}
}

func TestReasoningBudgetReservesAnswerSpace(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true})
	if err != nil {
		t.Fatal(err)
	}
	longThought := make([]int, 150)
	for i := range longThought {
		longThought[i] = 'a'
	}
	e := &budgetEngine{segments: [][]int{longThought, {'z'}}}
	_, err = generate(context.Background(), e, tok, parser, []int{'H'}, 200, engine.Sampling{}, nil, 500,
		func([]outputparser.Event) error { return nil })
	if err != nil || len(e.prompts) != 2 || len(e.prompts[1]) <= 101 {
		t.Fatalf("prompts=%v err=%v", e.prompts, err)
	}
	for _, id := range e.prompts[1][1:101] {
		if id != 'a' {
			t.Fatalf("budget not capped at 100 thought tokens: %v", e.prompts[1])
		}
	}
	if e.prompts[1][101] == 'a' {
		t.Fatalf("reasoning exceeded capped budget: %v", e.prompts[1])
	}
}

func TestReasoningBudgetStopInContinuedAnswer(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true})
	if err != nil {
		t.Fatal(err)
	}
	e := &budgetEngine{segments: [][]int{{'a', 'b', 'c', 'd'}, {'z', '!', 'x'}}}
	var content strings.Builder
	out, err := generate(context.Background(), e, tok, parser, []int{'H'}, 200, engine.Sampling{}, nil, 3,
		func(events []outputparser.Event) error {
			for _, event := range events {
				if event.Kind == "content" {
					if event.Text == "!" {
						return stopGeneration
					}
					content.WriteString(event.Text)
				}
			}
			return nil
		})
	if err != nil || len(e.prompts) != 2 || !out.stopped || content.String() != "z" {
		t.Fatalf("out=%+v prompts=%v content=%q err=%v", out, e.prompts, content.String(), err)
	}
}

func TestReasoningBudgetStopsAndContinues(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true})
	if err != nil {
		t.Fatal(err)
	}
	e := &budgetEngine{segments: [][]int{{'a', 'b', 'c', 'd'}, {'d', 'o', 'n', 'e'}}}
	prompt := []int{'H', 'i'}
	var reasoning, content strings.Builder
	out, err := generate(context.Background(), e, tok, parser, prompt, 200, engine.Sampling{}, nil, 3, func(events []outputparser.Event) error {
		for _, event := range events {
			if event.Kind == "reasoning" {
				reasoning.WriteString(event.Text)
			} else if event.Kind == "content" {
				content.WriteString(event.Text)
			}
		}
		return nil
	})
	extra, encodeErr := tok.Encode(reasoningWrapUp, true, nil)
	wantPrompt := append(append(slices.Clone(prompt), 'a', 'b', 'c'), extra...)
	if err != nil || encodeErr != nil || len(e.prompts) != 2 || !slices.Equal(e.prompts[1], wantPrompt) ||
		e.limits[1] != 200-3-len(extra) || content.String() != "done" ||
		reasoning.String() != "abc"+strings.Split(reasoningWrapUp, "</think>")[0] ||
		out.count != 3+len(extra)+4 || out.result.Finish != "stop" {
		t.Fatalf("out=%+v prompts=%v limits=%v reasoning=%q content=%q err=%v encode=%v",
			out, e.prompts, e.limits, reasoning.String(), content.String(), err, encodeErr)
	}
}

func TestReasoningBudgetWithoutRoomEndsAtLength(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true})
	if err != nil {
		t.Fatal(err)
	}
	e := &budgetEngine{segments: [][]int{{'a', 'a', 'a', 'a'}}}
	out, err := generate(context.Background(), e, tok, parser, []int{'H'}, 10, engine.Sampling{}, nil, 3,
		func([]outputparser.Event) error { return nil })
	if err != nil || len(e.prompts) != 1 || out.count != 3 || out.result.Finish != "length" {
		t.Fatalf("out=%+v prompts=%v err=%v", out, e.prompts, err)
	}
}
