package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"strata/engine"
	"strata/requestnorm"
)

type toolChoiceRenderer struct {
	mu     sync.Mutex
	tools  int
	prompt string
}

func TestToolChoiceNoneDoesNotEmitToolCall(t *testing.T) {
	h, _ := setupEngine(t, "engine-tool")
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":100,` +
		`"chat_template_kwargs":{"enable_thinking":false},"tool_choice":"none",` +
		`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/chat/completions", body, "secret")
	var result struct {
		Choices []struct {
			Finish  string `json:"finish_reason"`
			Message struct {
				Content string `json:"content"`
				Calls   []any  `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(raw), &result); status != http.StatusOK || err != nil ||
		len(result.Choices) != 1 || result.Choices[0].Finish != "stop" ||
		len(result.Choices[0].Message.Calls) != 0 || !strings.Contains(result.Choices[0].Message.Content, "lookup") {
		t.Fatalf("status %d, response %s, error %v", status, raw, err)
	}
}

func (r *toolChoiceRenderer) record(tools []requestnorm.Value) Rendered {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools = len(tools)
	r.prompt = "offered"
	if len(tools) == 0 {
		r.prompt = "omitted"
	}
	return Rendered{Prompt: r.prompt, Thinking: true}
}

func (r *toolChoiceRenderer) Render(_ context.Context, input requestnorm.Result) (Rendered, error) {
	return r.record(input.Tools), nil
}

func (r *toolChoiceRenderer) RenderAnthropic(_ context.Context, input requestnorm.AnthropicResult) (Rendered, error) {
	return r.record(input.Tools), nil
}

func TestToolChoiceHTTP(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tokenizer, _ := syntheticTokenizer(t)
	eng, err := engine.Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "engine"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = eng.Close(ctx)
	})
	renderer := &toolChoiceRenderer{}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tokenizer, Engine: eng,
		Renderer: renderer, APIKeys: []string{"secret"}, MaxBody: 4096, MaxPromptBytes: 2048,
		MaxOutputTokens: 200, MaxConcurrent: 2})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	defer ts.Close()
	for _, api := range []struct {
		name, path, prefix string
		tools              string
	}{
		{"openai", "/v1/chat/completions", `{"messages":[{"role":"user","content":"Hi"}],`,
			`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]`},
		{"anthropic", "/v1/messages", `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,`,
			`"tools":[{"name":"lookup","input_schema":{"type":"object"}}]`},
	} {
		for _, stream := range []bool{false, true} {
			streamField := ""
			if stream {
				streamField = `"stream":true,`
			}
			for _, choice := range []struct {
				name, value, wantPrompt string
				wantTools, status       int
			}{
				{"offered", "", "offered", 1, http.StatusOK},
				{"null", `"tool_choice":null,`, "offered", 1, http.StatusOK},
				{"none string", `"tool_choice":"none",`, "omitted", 0, http.StatusOK},
				{"none object", `"tool_choice":{"type":"none"},`, "omitted", 0, http.StatusOK},
				{"auto string", `"tool_choice":"auto",`, "offered", 1, http.StatusOK},
				{"auto object", `"tool_choice":{"type":"auto"},`, "offered", 1, http.StatusOK},
				{"required", `"tool_choice":"required",`, "offered", 1, http.StatusOK},
			} {
				t.Run(api.name+"/"+choice.name+"/stream="+map[bool]string{false: "false", true: "true"}[stream], func(t *testing.T) {
					body := api.prefix + streamField + choice.value + api.tools + `,"model":"test"}`
					status, response := request(t, ts.Client(), ts.URL+api.path, body, "secret")
					if status != choice.status {
						t.Fatalf("status %d, want %d: %s", status, choice.status, response)
					}
					if choice.status != http.StatusOK {
						return
					}
					if stream && !strings.Contains(response, "data:") && !strings.Contains(response, "event:") {
						t.Fatalf("stream response missing events: %s", response)
					}
					renderer.mu.Lock()
					gotTools, gotPrompt := renderer.tools, renderer.prompt
					renderer.mu.Unlock()
					if gotTools != choice.wantTools || gotPrompt != choice.wantPrompt {
						t.Fatalf("renderer saw tools=%d prompt=%q, want tools=%d prompt=%q",
							gotTools, gotPrompt, choice.wantTools, choice.wantPrompt)
					}
				})
			}
		}
	}
}
