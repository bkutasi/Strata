package server

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"strata/engine"
	"strata/outputparser"
	"strata/requestnorm"
)

func bytesAsTokens(text string) []int {
	result := make([]int, 0, len(text))
	for _, b := range []byte(text) {
		result = append(result, int(b))
	}
	return result
}

func TestForcedThinkingHeldClosingTag(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true, Tools: []map[string]any{{"name": "lookup"}}})
	if err != nil {
		t.Fatal(err)
	}
	first := append(bytesAsTokens(`thinking "quoted"</think>`), 257)
	e := &budgetEngine{segments: [][]int{first, bytesAsTokens("</function></tool_call>")}}
	var calls int
	out, err := generateWithForce(context.Background(), e, tok, parser, []int{'H'}, 300, engine.Sampling{},
		map[int]bool{257: true}, 0, "<tool_call>\n<function=lookup>\n", func(events []outputparser.Event) error {
			for _, event := range events {
				if event.Kind == "tool_call" {
					calls++
				}
			}
			return nil
		})
	if err != nil || len(e.prompts) != 2 || calls != 1 || out.stopped {
		t.Fatalf("out=%+v prompts=%v calls=%d err=%v", out, e.prompts, calls, err)
	}
}

func TestForcedThinkingHonorsModelToolCall(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true, Tools: []map[string]any{{"name": "lookup"}}})
	if err != nil {
		t.Fatal(err)
	}
	e := &budgetEngine{segments: [][]int{bytesAsTokens("<tool_call><function=lookup></function></tool_call></think>\n")}}
	var calls int
	_, err = generateWithForce(context.Background(), e, tok, parser, []int{'H'}, 300, engine.Sampling{},
		nil, 0, "<tool_call>\n<function=lookup>\n", func(events []outputparser.Event) error {
			for _, event := range events {
				if event.Kind == "tool_call" {
					calls++
				}
			}
			return nil
		})
	if err != nil || len(e.prompts) != 1 || calls != 1 {
		t.Fatalf("prompts=%v calls=%d err=%v", e.prompts, calls, err)
	}
}

func TestDeepSeekForcedThinkingBudget(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.NewDeepSeek(outputparser.Options{Thinking: true, Tools: []map[string]any{{"name": "lookup"}}})
	if err != nil {
		t.Fatal(err)
	}
	e := &budgetEngine{segments: [][]int{
		bytesAsTokens("thought"),
		bytesAsTokens(`<｜DSML｜ parameter name="q" string="true">yes</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`),
	}}
	var calls int
	_, err = generateWithForce(context.Background(), e, tok, parser, []int{'H'}, 300, engine.Sampling{}, nil, 3,
		deepSeekForcePrefix("<tool_call>\n<function=lookup>\n"), func(events []outputparser.Event) error {
			for _, event := range events {
				if event.Kind == "tool_call" && event.Call.Name == "lookup" && event.Call.Arguments["q"] == "yes" {
					calls++
				}
			}
			return nil
		})
	if err != nil || len(e.prompts) != 2 || calls != 1 {
		t.Fatalf("prompts=%v calls=%d err=%v", e.prompts, calls, err)
	}
}

func TestForcedToolWithThinking(t *testing.T) {
	for _, api := range []struct{ path, choice, tools string }{
		{"/v1/chat/completions", `"tool_choice":{"type":"function","function":{"name":"lookup"}}`, `"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]`},
		{"/v1/messages", `"tool_choice":{"type":"tool","name":"lookup"}`, `"tools":[{"name":"lookup","input_schema":{"type":"object"}}]`},
	} {
		for _, budget := range []bool{false, true} {
			for _, stream := range []bool{false, true} {
				tok, _ := syntheticTokenizer(t)
				eng := &budgetEngine{segments: [][]int{
					bytesAsTokens("thought</think>\nignored"),
					bytesAsTokens("<parameter=q>yes</parameter></function></tool_call>"),
				}}
				h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: eng,
					Renderer: &toolChoiceRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
				if err != nil {
					t.Fatal(err)
				}
				body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,` + api.choice + `,` + api.tools
				if budget {
					body += `,"reasoning_budget_tokens":3`
				}
				if stream {
					body += `,"stream":true`
				}
				w := responsesRequest(h, "http://127.0.0.1"+api.path, body+`}`, "")
				if w.Code != http.StatusOK || len(eng.prompts) != 2 ||
					!strings.Contains(w.Body.String(), "lookup") || !strings.Contains(w.Body.String(), "yes") {
					t.Fatalf("%s budget=%v stream=%v: status=%d prompts=%v body=%s", api.path, budget, stream, w.Code, eng.prompts, w.Body)
				}
				prompt, err := tok.Decode(eng.prompts[1])
				if err != nil || !strings.HasSuffix(prompt, "<tool_call>\n<function=lookup>\n") {
					t.Fatalf("continuation prompt %q: %v", prompt, err)
				}
			}
		}
	}
}
