package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"strata/requestnorm"
)

func TestJSONSchemaOutput(t *testing.T) {
	for _, api := range []struct {
		path, prefix, suffix string
	}{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":`, `}}`},
		{"/v1/responses", `{"input":"hi","text":{"format":`, `}}`},
	} {
		for _, tc := range []struct {
			name, schema, output string
			status               int
		}{
			{"valid", `{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}`, `{"n":3}`, http.StatusOK},
			{"invalid output", `{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}`, `{"n":"bad"}`, http.StatusBadGateway},
			{"missing required", `{"type":"object","required":["n"]}`, `{}`, http.StatusBadGateway},
			{"invalid schema", `{"type":"object","properties":{"n":{"type":"invalid"}}}`, `{}`, http.StatusBadRequest},
			{"remote reference", `{"type":"object","$ref":"file:///etc/passwd"}`, `{}`, http.StatusBadRequest},
			{"non-object root", `{"type":"array"}`, `{}`, http.StatusBadRequest},
			{"local reference", `{"$ref":"#/$defs/x","$defs":{"x":{"type":"object","required":["n"]}}}`, `{"n":3}`, http.StatusOK},
			{"cyclic reference", `{"$ref":"#/$defs/x","$defs":{"x":{"$ref":"#/$defs/x"}}}`, `{}`, http.StatusBadRequest},
			{"object union", `{"anyOf":[{"type":"object"},{"type":"object"}]}`, `{}`, http.StatusOK},
			{"mixed union", `{"anyOf":[{"type":"object"},{"type":"string"}]}`, `{}`, http.StatusBadRequest},
		} {
			t.Run(api.path+"/"+tc.name, func(t *testing.T) {
				for _, stream := range []bool{false, true} {
					h := responsesHandler(t, tc.output, "stop", false)
					spec := `{"name":"test","schema":` + tc.schema + `}`
					if api.path == "/v1/responses" {
						spec = `{"type":"json_schema","name":"test","schema":` + tc.schema + `}`
					}
					body := api.prefix + spec + api.suffix
					if stream {
						body = strings.TrimSuffix(body, "}") + `,"stream":true}`
					}
					w := responsesRequest(h, api.path, body, "secret")
					status := tc.status
					if stream && status == http.StatusBadGateway {
						status = http.StatusOK
					}
					if w.Code != status {
						t.Fatalf("stream=%v: status %d, want %d: %s", stream, w.Code, status, w.Body.String())
					}
					if tc.status == http.StatusBadGateway && !strings.Contains(w.Body.String(), `"structured_output_failed"`) {
						t.Fatalf("stream=%v: missing structured output error: %s", stream, w.Body.String())
					}
					if stream && tc.status == http.StatusBadGateway &&
						(strings.Contains(w.Body.String(), `"delta"`) || strings.Contains(w.Body.String(), "event: response.output_text.delta")) {
						t.Fatalf("stream leaked invalid structured content: %s", w.Body.String())
					}
				}
			})
		}
	}
}

type schemaRenderer struct{ input requestnorm.Result }

func (r *schemaRenderer) Render(_ context.Context, input requestnorm.Result) (Rendered, error) {
	r.input = input
	return Rendered{Prompt: "Hi"}, nil
}

func TestJSONSchemaReachesPrompt(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	renderer := &schemaRenderer{}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: responsesEngine{text: `{"n":3}`, finish: "stop"}, Renderer: renderer,
		APIKeys: []string{"secret"}, MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	schema := `{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}`
	body := `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema",` +
		`"json_schema":{"name":"test","schema":` + schema + `}}}`
	w := responsesRequest(h, "/v1/chat/completions", body, "secret")
	if w.Code != http.StatusOK || len(renderer.input.Messages) == 0 ||
		!strings.Contains(renderer.input.Messages[0].Content.Text, schema) {
		t.Fatalf("status %d, prompt input %+v: %s", w.Code, renderer.input.Messages, w.Body.String())
	}
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil || len(result.Choices) != 1 ||
		result.Choices[0].Message.Content != `{"n":3}` {
		t.Fatalf("response %s, error %v", w.Body.String(), err)
	}
}

func TestJSONSchemaRejectsInvalidSpecifications(t *testing.T) {
	h := responsesHandler(t, `{}`, "stop", false)
	for _, spec := range []string{
		`{"schema":{"type":"object"}}`,
		`{"name":"","schema":{"type":"object"}}`,
		`{"name":"x","strict":null,"schema":{"type":"object"}}`,
		`{"name":"x","strict":"true","schema":{"type":"object"}}`,
		`{"name":"x","schema":{"type":"object","properties":{"x":{"$ref":"https://example.test/schema"}}}}`,
		`{"name":"x","schema":{"type":"object","$schema":"file:///etc/passwd"}}`,
	} {
		body := `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":` + spec + `}}`
		w := responsesRequest(h, "/v1/chat/completions", body, "secret")
		if w.Code != http.StatusBadRequest {
			t.Errorf("spec %s: status %d: %s", spec, w.Code, w.Body.String())
		}
	}
}

func TestResponsesJSONSchemaAllowsToolCall(t *testing.T) {
	h := responsesHandler(t, `<tool_call><function=lookup></function></tool_call>`, "stop", false)
	body := `{"input":"hello","text":{"format":{"type":"json_schema","name":"answer",` +
		`"schema":{"type":"object","required":["n"]}}},` +
		`"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`
	for _, stream := range []bool{false, true} {
		request := body
		if stream {
			request = strings.TrimSuffix(body, "}") + `,"stream":true}`
		}
		w := responsesRequest(h, "/v1/responses", request, "secret")
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"type":"function_call"`) ||
			strings.Contains(w.Body.String(), "structured_output_failed") {
			t.Errorf("stream=%v: %d %s", stream, w.Code, w.Body.String())
		}
	}
}
