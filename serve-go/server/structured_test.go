package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestStructuredObjectValidation(t *testing.T) {
	for _, tc := range []struct {
		input, finish, want string
		repair              bool
		bad                 bool
	}{
		{`Here: {"b": 1, "a": 1.0} done`, "stop", `{"b":1,"a":1.0}`, false, false},
		{"```json\n{\"x\":\"a } b\"}\n```", "stop", `{"x":"a } b"}`, false, false},
		{`{"x":1e-5,"y":1e16}`, "stop", `{"x":1e-05,"y":1e+16}`, false, false},
		{`{"a":"\alpha"}`, "stop", `{"a":"\\alpha"}`, true, false},
		{`{"a":"\u+123"}`, "stop", `{"a":"\\u+123"}`, true, false},
		{`{"a":"\u00e9","b":"a\n"}`, "stop", `{"a":"é","b":"a\n"}`, true, false},
		{`{"\u0001":1}`, "stop", `{"\u0001":1}`, false, false},
		{`{"a":"\alpha"}`, "stop", "", false, true},
		{`{"x":1,"x":2}`, "stop", "", false, true},
		{`{"x":{"y":1,"y":2}}`, "stop", "", false, true},
		{`{"x":1e999}`, "stop", "", false, true},
		{`[]`, "stop", "", false, true},
		{`{"x":`, "stop", "", false, true},
		{`{"x":1}`, "length", "", false, true},
	} {
		got, err := validateJSONObject(tc.input, tc.finish, tc.repair)
		if got != tc.want || (err != nil) != tc.bad {
			t.Errorf("%s: got %q, %v; want %q, bad=%v", tc.input, got, err, tc.want, tc.bad)
		}
	}
}

func TestChatJSONObjectToolChoice(t *testing.T) {
	h := responsesHandler(t, "{}", "stop", false)
	body := `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"},` +
		`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]`
	if w := responsesRequest(h, "/v1/chat/completions", body+`}`, "secret"); w.Code != http.StatusBadRequest {
		t.Fatalf("with tools: %d %s", w.Code, w.Body.String())
	}
	if w := responsesRequest(h, "/v1/chat/completions", body+`,"tool_choice":"none"}`, "secret"); w.Code != http.StatusOK {
		t.Fatalf("without tools: %d %s", w.Code, w.Body.String())
	}
}

func TestChatJSONObjectEscapeRepair(t *testing.T) {
	for _, tc := range []struct {
		repair bool
		status int
	}{
		{false, http.StatusBadGateway},
		{true, http.StatusOK},
	} {
		h := responsesHandler(t, `{"value":"\alpha"}`, "stop", false)
		body := `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object","repair_latex_escapes":`
		if tc.repair {
			body += `true`
		} else {
			body += `false`
		}
		w := responsesRequest(h, "/v1/chat/completions", body+`}}`, "secret")
		if w.Code != tc.status || (tc.repair && !strings.Contains(w.Body.String(), `\\\\alpha`)) {
			t.Errorf("repair=%v: %d %s", tc.repair, w.Code, w.Body.String())
		}
	}
}

func TestChatJSONObject(t *testing.T) {
	for _, tc := range []struct {
		text, finish string
		want         string
		bad          bool
	}{
		{`answer: {"x": 1} done`, "stop", `{"x":1}`, false},
		{`{"x":1,"x":2}`, "stop", "", true},
		{`{"x":1}`, "length", "", true},
	} {
		for _, stream := range []bool{false, true} {
			h := responsesHandler(t, tc.text, tc.finish, false)
			body := `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}`
			if stream {
				body += `,"stream":true`
			}
			w := responsesRequest(h, "/v1/chat/completions", body+`}`, "secret")
			if tc.bad {
				if stream {
					if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"structured_output_failed"`) ||
						strings.Contains(w.Body.String(), `"delta"`) || !strings.Contains(w.Body.String(), "data: [DONE]") {
						t.Errorf("invalid stream: %d %s", w.Code, w.Body.String())
					}
				} else if w.Code != http.StatusBadGateway || !strings.Contains(w.Body.String(), `"structured_output_failed"`) {
					t.Errorf("invalid response: %d %s", w.Code, w.Body.String())
				}
				continue
			}
			var content string
			if stream {
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if !strings.HasPrefix(line, "data: {") {
						continue
					}
					var chunk struct {
						Choices []struct {
							Delta struct {
								Content string `json:"content"`
							} `json:"delta"`
						} `json:"choices"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
						t.Fatal(err)
					}
					if len(chunk.Choices) != 0 {
						content += chunk.Choices[0].Delta.Content
					}
				}
			} else {
				var response struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || len(response.Choices) != 1 {
					t.Fatalf("status %d: %s, %v", w.Code, w.Body.String(), err)
				}
				content = response.Choices[0].Message.Content
			}
			if w.Code != http.StatusOK || content != tc.want {
				t.Errorf("stream=%v: status %d, content %q, want %q", stream, w.Code, content, tc.want)
			}
		}
	}
}
