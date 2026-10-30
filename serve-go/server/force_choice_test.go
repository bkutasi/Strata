package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
)

func TestForcedToolPrefixes(t *testing.T) {
	parse := func(raw string) requestnorm.Value {
		t.Helper()
		v, err := requestnorm.ParseValue([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	one := []requestnorm.Value{parse(`{"name":"search"}`)}
	two := append(append([]requestnorm.Value(nil), one...), parse(`{"name":"read"}`))
	for _, tc := range []struct {
		choice string
		tools  []requestnorm.Value
		family requestnorm.Family
		kind   string
		prefix string
	}{
		{`null`, one, requestnorm.FamilyQwen, "auto", ""},
		{`"none"`, one, requestnorm.FamilyQwen, "none", ""},
		{`"required"`, one, requestnorm.FamilyQwen, "named", "<tool_call>\n<function=search>\n"},
		{`"required"`, two, requestnorm.FamilyQwen, "required", "<tool_call>\n<function="},
		{`{"type":"function","function":{"name":"read"}}`, two, requestnorm.FamilyQwen, "named", "<tool_call>\n<function=read>\n"},
		{`{"type":"any"}`, one, requestnorm.FamilyDeepSeek, "named", "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search\">\n"},
		{`{"type":"any"}`, two, requestnorm.FamilyDeepSeek, "required", "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\""},
		{`{"type":"tool","name":"read"}`, two, requestnorm.FamilyDeepSeek, "named", "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n"},
		{`"required"`, nil, requestnorm.FamilyQwen, "auto", ""},
		{`{"type":"tool","name":"missing"}`, two, requestnorm.FamilyQwen, "auto", ""},
		{`{"type":"other"}`, one, requestnorm.FamilyQwen, "auto", ""},
	} {
		gotKind, gotPrefix := normalizeForceChoice(parse(tc.choice), tc.tools, tc.family)
		if gotKind != tc.kind || gotPrefix != tc.prefix {
			t.Errorf("%s %s: got (%q, %q), want (%q, %q)", tc.family, tc.choice, gotKind, gotPrefix, tc.kind, tc.prefix)
		}
	}
}

type forceEngine struct {
	tok    *modeltokenizer.Tokenizer
	prompt string
	reply  string
}

func (*forceEngine) Alive() bool       { return true }
func (*forceEngine) ContextLimit() int { return 4096 }
func (e *forceEngine) GenerateWithOptions(_ context.Context, ids []int, _ int, opts engine.RequestOptions) (engine.Result, error) {
	var err error
	e.prompt, err = e.tok.Decode(ids)
	if err != nil {
		return engine.Result{}, err
	}
	for _, b := range []byte(e.reply) {
		if err := opts.OnToken(int(b)); err != nil {
			return engine.Result{}, err
		}
	}
	return engine.Result{Finish: "stop"}, nil
}

func TestForcedToolsWithoutThinking(t *testing.T) {
	for _, family := range []requestnorm.Family{requestnorm.FamilyQwen, requestnorm.FamilyDeepSeek} {
		for _, api := range []string{"chat", "anthropic"} {
			t.Run(string(family)+"/"+api, func(t *testing.T) {
				tok, _ := syntheticTokenizer(t)
				prefix := "<tool_call>\n<function=lookup>\n"
				reply := "<parameter=query>apples</parameter></function></tool_call>"
				if family == requestnorm.FamilyDeepSeek {
					prefix = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n"
					reply = `<｜DSML｜ parameter name="query" string="true">apples</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`
				}
				eng := &forceEngine{tok: tok, reply: reply}
				handler, err := New(Config{Model: "test", Family: family, Tokenizer: tok, Engine: eng,
					Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
					MaxOutputTokens: 512, MaxConcurrent: 1})
				if err != nil {
					t.Fatal(err)
				}
				path := "/v1/chat/completions"
				body := `{"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]}`
				if api == "anthropic" {
					path = "/v1/messages"
					body = `{"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"tool","name":"lookup"},"tools":[{"name":"lookup","input_schema":{"type":"object","properties":{"query":{"type":"string"}}}}]}`
				}
				w := responsesRequest(handler, "http://127.0.0.1"+path, body, "")
				if w.Code != http.StatusOK || eng.prompt != "Hi"+prefix {
					t.Fatalf("status %d, prompt %q: %s", w.Code, eng.prompt, w.Body.String())
				}
				if api == "chat" {
					var result struct {
						Choices []struct {
							Finish  string `json:"finish_reason"`
							Message struct {
								Calls []struct {
									Function struct {
										Name      string `json:"name"`
										Arguments string `json:"arguments"`
									} `json:"function"`
								} `json:"tool_calls"`
							} `json:"message"`
						} `json:"choices"`
					}
					if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil || len(result.Choices) != 1 ||
						result.Choices[0].Finish != "tool_calls" || len(result.Choices[0].Message.Calls) != 1 {
						t.Fatalf("invalid chat result: %s, %v", w.Body.String(), err)
					}
					call := result.Choices[0].Message.Calls[0].Function
					var arguments map[string]string
					if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil || call.Name != "lookup" || arguments["query"] != "apples" {
						t.Fatalf("invalid call: %+v, %v", call, err)
					}
				} else {
					var result struct {
						Content []struct {
							Type  string            `json:"type"`
							Name  string            `json:"name"`
							Input map[string]string `json:"input"`
						} `json:"content"`
					}
					if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil || len(result.Content) != 1 ||
						result.Content[0].Type != "tool_use" || result.Content[0].Name != "lookup" || result.Content[0].Input["query"] != "apples" {
						t.Fatalf("invalid Anthropic result: %s, %v", w.Body.String(), err)
					}
				}
				eng.prompt = ""
				streamBody := strings.TrimSuffix(body, "}") + `,"stream":true}`
				stream := responsesRequest(handler, "http://127.0.0.1"+path, streamBody, "")
				if stream.Code != http.StatusOK || eng.prompt != "Hi"+prefix ||
					!strings.Contains(stream.Body.String(), "lookup") || !strings.Contains(stream.Body.String(), "apples") {
					t.Fatalf("stream status %d, prompt %q: %s", stream.Code, eng.prompt, stream.Body.String())
				}
			})
		}
	}
}

func TestRequiredChoiceWithMultipleTools(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	eng := &forceEngine{tok: tok, reply: "lookup>\n<parameter=query>apples</parameter></function></tool_call>"}
	handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: eng,
		Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"messages":[{"role":"user","content":"hi"}],"tool_choice":"required","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}},{"type":"function","function":{"name":"other","parameters":{"type":"object"}}}]}`
	w := responsesRequest(handler, "http://127.0.0.1/v1/chat/completions", body, "")
	if w.Code != http.StatusOK || eng.prompt != "Hi<tool_call>\n<function=" {
		t.Fatalf("status %d, prompt %q: %s", w.Code, eng.prompt, w.Body.String())
	}
	var response struct {
		Choices []struct {
			Finish  string `json:"finish_reason"`
			Message struct {
				Calls []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || len(response.Choices) != 1 ||
		response.Choices[0].Finish != "tool_calls" || len(response.Choices[0].Message.Calls) != 1 ||
		response.Choices[0].Message.Calls[0].Function.Name != "lookup" {
		t.Fatalf("invalid required-tool response: %s, %v", w.Body.String(), err)
	}
}

func TestForcedPromptRespectsLimits(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	prompt := Rendered{Prompt: "Hi"}
	for _, tc := range []struct {
		rendered Rendered
		limit    int
	}{
		{prompt, 2},
	} {
		if _, err := appendForcedPrompt(tok, tc.rendered, []int{'H', 'i'}, "x", tc.limit); err == nil {
			t.Fatalf("forced prompt unexpectedly accepted: %+v", tc)
		}
	}
	ids, err := appendForcedPrompt(tok, prompt, []int{'H', 'i'}, "x", 3)
	if err != nil || len(ids) != 3 || ids[2] != 'x' {
		t.Fatalf("forced prompt = %v, %v", ids, err)
	}
}
