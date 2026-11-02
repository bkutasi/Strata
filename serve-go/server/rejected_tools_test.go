package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type rejectedToolEngine struct{ text string }

func (rejectedToolEngine) Alive() bool       { return true }
func (rejectedToolEngine) ContextLimit() int { return 4096 }
func (e rejectedToolEngine) GenerateWithOptions(_ context.Context, _ []int, _ int, o engine.RequestOptions) (engine.Result, error) {
	for _, b := range []byte(e.text) {
		if err := o.OnToken(int(b)); err != nil {
			return engine.Result{}, err
		}
	}
	return engine.Result{Finish: "stop"}, nil
}

func TestRejectedToolMetadata(t *testing.T) {
	const rejected = "I will check.\n<tool_call>\n<function=bash>\n<parameter=q>\n2+2\n</parameter>\n</function>\n</tool_call>"
	for _, tc := range []struct {
		name, path, tools string
	}{
		{"chat", "/v1/chat/completions", `"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]`},
		{"anthropic", "/v1/messages", `"tools":[{"name":"search","input_schema":{"type":"object"}}]`},
	} {
		for _, stream := range []bool{false, true} {
			for _, text := range []string{rejected, "ordinary answer"} {
				t.Run(tc.name+"/"+map[bool]string{false: "one-shot", true: "stream"}[stream]+"/"+map[bool]string{false: "normal", true: "rejected"}[text == rejected], func(t *testing.T) {
					tok := sharedTokenizer(t)
					h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
						Engine: rejectedToolEngine{text}, Renderer: contextRenderer{}, MaxBody: 8192,
						MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
					if err != nil {
						t.Fatal(err)
					}
					ts := httptest.NewServer(h)
					defer ts.Close()
					body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"stream":` +
						map[bool]string{false: "false", true: "true"}[stream] + `,` + tc.tools + `}`
					status, raw := request(t, ts.Client(), ts.URL+tc.path, body, "")
					if status != http.StatusOK {
						t.Fatalf("status=%d body=%s", status, raw)
					}
					var replies []map[string]any
					if !stream {
						var value map[string]any
						if err := json.Unmarshal([]byte(raw), &value); err != nil {
							t.Fatal(err)
						}
						replies = append(replies, value)
					} else {
						for _, line := range strings.Split(raw, "\n") {
							if !strings.HasPrefix(line, "data: {") {
								continue
							}
							var value map[string]any
							if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &value); err != nil {
								t.Fatal(err)
							}
							replies = append(replies, value)
						}
					}
					found := 0
					var content strings.Builder
					for _, value := range replies {
						if choices, ok := value["choices"].([]any); ok && len(choices) > 0 {
							choice, _ := choices[0].(map[string]any)
							part, _ := choice["message"].(map[string]any)
							if stream {
								part, _ = choice["delta"].(map[string]any)
							}
							if text, ok := part["content"].(string); ok {
								content.WriteString(text)
							}
						}
						if blocks, ok := value["content"].([]any); ok {
							for _, block := range blocks {
								if part, ok := block.(map[string]any); ok {
									if text, ok := part["text"].(string); ok {
										content.WriteString(text)
									}
								}
							}
						}
						if delta, ok := value["delta"].(map[string]any); ok {
							if text, ok := delta["text"].(string); ok {
								content.WriteString(text)
							}
						}
						if marker, ok := value["strata_rejected_tool_calls"]; ok {
							found++
							data, err := json.Marshal(marker)
							if err != nil || string(data) != `[{"name":"bash","reason":"undeclared"}]` {
								t.Fatalf("marker=%s error=%v", data, err)
							}
							if stream && tc.name == "anthropic" && value["type"] != "message_delta" {
								t.Fatalf("marker not on final message_delta: %v", value)
							}
							if stream && tc.name == "chat" {
								choices, ok := value["choices"].([]any)
								if !ok || len(choices) != 1 {
									t.Fatalf("marker without final chat choice: %v", value)
								}
								choice, ok := choices[0].(map[string]any)
								if !ok || choice["finish_reason"] != "stop" {
									t.Fatalf("marker not on final chat chunk: %v", value)
								}
							}
						}
					}
					want := 0
					if text == rejected {
						want = 1
					}
					if found != want {
						t.Fatalf("markers=%d want=%d body=%s", found, want, raw)
					}
					if text == rejected && !strings.Contains(content.String(), "<function=bash>") {
						t.Fatalf("rejected call text missing from content: %q", content.String())
					}
				})
			}
		}
	}
}
