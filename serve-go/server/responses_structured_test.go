package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestResponsesJSONObject(t *testing.T) {
	for _, tc := range []struct {
		name, text, finish, want string
		repair, bad              bool
	}{
		{"valid", `Here: {"x": 1}`, "stop", `{"x":1}`, false, false},
		{"repair", `{"x":"\alpha"}`, "stop", `{"x":"\\alpha"}`, true, false},
		{"duplicate", `{"x":1,"x":2}`, "stop", "", false, true},
		{"length", `{"x":1}`, "length", "", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, stream := range []bool{false, true} {
				h := responsesHandler(t, tc.text, tc.finish, false)
				body := `{"input":"hello","text":{"format":{"type":"json_object"`
				if tc.repair {
					body += `,"repair_latex_escapes":true`
				}
				body += `}}`
				if stream {
					body += `,"stream":true`
				}
				w := responsesRequest(h, "/v1/responses", body+`}`, "secret")
				if tc.bad {
					if stream {
						if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "event: response.failed\n") ||
							!strings.Contains(w.Body.String(), `"structured_output_failed"`) ||
							strings.Contains(w.Body.String(), "event: response.output_text.delta\n") {
							t.Errorf("invalid stream: %d %s", w.Code, w.Body.String())
						}
					} else if w.Code != http.StatusBadGateway || !strings.Contains(w.Body.String(), `"structured_output_failed"`) {
						t.Errorf("invalid response: %d %s", w.Code, w.Body.String())
					}
					continue
				}
				raw := w.Body.String()
				if stream {
					for _, line := range strings.Split(raw, "\n") {
						if strings.HasPrefix(line, "data: ") && strings.Contains(line, `"type":"response.completed"`) {
							raw = strings.TrimPrefix(line, "data: ")
							break
						}
					}
					var event struct {
						Response json.RawMessage `json:"response"`
					}
					if err := json.Unmarshal([]byte(raw), &event); err != nil {
						t.Fatal(err)
					}
					raw = string(event.Response)
				}
				var response struct {
					Status string `json:"status"`
					Output []struct {
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"output"`
				}
				if err := json.Unmarshal([]byte(raw), &response); err != nil || w.Code != http.StatusOK ||
					response.Status != "completed" || len(response.Output) != 1 ||
					len(response.Output[0].Content) != 1 || response.Output[0].Content[0].Text != tc.want {
					t.Errorf("stream=%v: status %d: %s; %v", stream, w.Code, w.Body.String(), err)
				}
			}
		})
	}
}

func TestResponsesJSONObjectAllowsToolCall(t *testing.T) {
	h := responsesHandler(t, `before<tool_call><function=lookup></function></tool_call>after`, "stop", false)
	body := `{"input":"hello","text":{"format":{"type":"json_object"}},` +
		`"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`
	for _, stream := range []bool{false, true} {
		request := body
		if stream {
			request = strings.TrimSuffix(body, "}") + `,"stream":true}`
		}
		w := responsesRequest(h, "/v1/responses", request, "secret")
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"type":"function_call"`) ||
			!strings.Contains(w.Body.String(), "before") || !strings.Contains(w.Body.String(), "after") ||
			strings.Contains(w.Body.String(), "structured_output_failed") {
			t.Errorf("stream=%v: %d %s", stream, w.Code, w.Body.String())
		}
	}
}

func TestResponsesJSONObjectPreservesReasoning(t *testing.T) {
	h := responsesHandler(t, `thought</think>{"value":1}`, "stop", true)
	body := `{"input":"hello","text":{"format":{"type":"json_object"}}}`
	w := responsesRequest(h, "/v1/responses", body, "secret")
	var response struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusOK ||
		len(response.Output) != 2 || response.Output[0].Type != "reasoning" ||
		response.Output[0].Content[0].Text != "thought" || response.Output[1].Content[0].Text != `{"value":1}` {
		t.Fatalf("%d %s: %v", w.Code, w.Body.String(), err)
	}
}
