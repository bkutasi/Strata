package server

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type contextEngine struct {
	limit, requested, calls int
}

func (e *contextEngine) Alive() bool       { return true }
func (e *contextEngine) ContextLimit() int { return e.limit }
func (e *contextEngine) GenerateWithOptions(_ context.Context, _ []int, max int, opts engine.RequestOptions) (engine.Result, error) {
	e.calls++
	e.requested = max
	if err := opts.OnToken(int('A')); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop"}, nil
}

type contextRenderer struct{}

func (contextRenderer) Render(context.Context, requestnorm.Result) (Rendered, error) {
	return Rendered{Prompt: "Hi"}, nil
}
func (contextRenderer) RenderAnthropic(context.Context, requestnorm.AnthropicResult) (Rendered, error) {
	return Rendered{Prompt: "Hi"}, nil
}

func TestFitOutputTokens(t *testing.T) {
	for _, tc := range []struct {
		context, prompt, requested, want int
		ok                               bool
	}{
		{13, 2, 2, 2, true},
		{13, 2, 3, 3, true},
		{13, 2, 100, 3, true},
		{10, 2, 1, 0, false},
		{9, 2, 1, 0, false},
	} {
		got, err := fitOutputTokens(tc.context, tc.prompt, tc.requested, false, false)
		if got != tc.want || (err == nil) != tc.ok {
			t.Errorf("fit(%d, %d, %d) = %d, %v", tc.context, tc.prompt, tc.requested, got, err)
		}
	}
}

func TestStrictOutputLimitAcrossAPIs(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &contextEngine{limit: 13}
	handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 512, MaxConcurrent: 1, DisableFitMaxTokens: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, body, limit string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}`, "max_tokens"},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}`, "max_tokens"},
		{"/v1/responses", `{"input":"hi","max_output_tokens":4}`, "max_output_tokens"},
	} {
		for _, stream := range []bool{false, true} {
			body := strings.TrimSuffix(tc.body, "}")
			if stream {
				body += `,"stream":true}`
			} else {
				body += "}"
			}
			eng.limit, eng.requested = 13, 0
			before := eng.calls
			response := responsesRequest(handler, "http://127.0.0.1"+tc.path, body, "")
			if response.Code != http.StatusBadRequest || eng.calls != before {
				t.Errorf("%s stream=%v: status %d, generation calls %d, body %s", tc.path, stream, response.Code, eng.calls-before, response.Body)
			}
			if !strings.Contains(response.Body.String(), "exceeds the context window") ||
				!strings.Contains(response.Body.String(), "prompt (2 tokens)") ||
				!strings.Contains(response.Body.String(), "context (13)") {
				t.Errorf("%s: missing recognizable overflow: %s", tc.path, response.Body)
			}
			if tc.path == "/v1/responses" && (!strings.Contains(response.Body.String(), `"code":"context_length_exceeded"`) || !strings.Contains(response.Body.String(), `"param":"input"`)) {
				t.Errorf("Responses error shape: %s", response.Body)
			}
		}
	}
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}`},
		{"/v1/responses", `{"input":"hi"}`},
		{"/v1/responses", `{"input":"hi","max_output_tokens":0}`},
	} {
		eng.limit, eng.requested = 13, 0
		response := responsesRequest(handler, "http://127.0.0.1"+tc.path, tc.body, "")
		if response.Code != http.StatusOK || eng.requested != 3 {
			t.Errorf("%s nonpositive/missing: status %d, requested %d, body %s", tc.path, response.Code, eng.requested, response.Body)
		}
		eng.limit = 10
		before := eng.calls
		response = responsesRequest(handler, "http://127.0.0.1"+tc.path, tc.body, "")
		if response.Code != http.StatusBadRequest || eng.calls != before || !strings.Contains(response.Body.String(), "no room to answer") {
			t.Errorf("%s nonpositive/missing: status %d, calls %d, body %s", tc.path, response.Code, eng.calls-before, response.Body)
		}
	}
	eng.limit = 10
	response := responsesRequest(handler, "http://127.0.0.1/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}`, "")
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "+ max tokens (4)") {
		t.Errorf("strict prompt overflow: status %d, body %s", response.Code, response.Body)
	}
}

func TestAPIsFitOutputToContext(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &contextEngine{limit: 13}
	handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`},
		{"/v1/responses", `{"input":"hi"}`},
	} {
		eng.limit, eng.requested = 13, 0
		response := responsesRequest(handler, "http://127.0.0.1"+tc.path, tc.body, "")
		if response.Code != http.StatusOK || eng.requested != 3 {
			t.Errorf("%s: status %d, requested %d: %s", tc.path, response.Code, eng.requested, response.Body)
		}
		eng.limit, eng.requested = 10, 0
		response = responsesRequest(handler, "http://127.0.0.1"+tc.path, tc.body, "")
		if response.Code != http.StatusBadRequest || eng.requested != 0 {
			t.Errorf("%s no room: status %d, requested %d: %s", tc.path, response.Code, eng.requested, response.Body)
		}
	}
}

func TestLargeOutputBudgetFitsContext(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &contextEngine{limit: 8192}
	handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: int(^uint(0) >> 1), MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		body string
		want int
	}{
		{`{"messages":[{"role":"user","content":"hi"}],"max_tokens":5000}`, 5000},
		{`{"messages":[{"role":"user","content":"hi"}]}`, 8182},
	} {
		eng.requested = 0
		response := responsesRequest(handler, "http://127.0.0.1/v1/chat/completions", tc.body, "")
		if response.Code != http.StatusOK || eng.requested != tc.want {
			t.Errorf("status %d, requested %d, want %d: %s", response.Code, eng.requested, tc.want, response.Body)
		}
	}
}

func TestNonPositiveOutputLimitsUseRemainingContext(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &contextEngine{limit: 100}
	handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 100, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}`},
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}`},
		{"/v1/responses", `{"input":"hi","max_output_tokens":0}`},
		{"/v1/responses", `{"input":"hi","max_output_tokens":-1}`},
	} {
		eng.requested = 0
		response := responsesRequest(handler, "http://127.0.0.1"+tc.path, tc.body, "")
		if response.Code != http.StatusOK || eng.requested != 90 {
			t.Errorf("%s %s: status %d, requested %d: %s", tc.path, tc.body, response.Code, eng.requested, response.Body)
		}
	}
}

func TestChatMaxCompletionTokensPrecedence(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &contextEngine{limit: 100}
	handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 100, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		fields string
		want   int
	}{
		{`"max_tokens":10,"max_completion_tokens":20`, 20},
		{`"max_completion_tokens":20,"max_tokens":10`, 20},
		{`"max_tokens":10,"max_completion_tokens":0`, 10},
		{`"max_tokens":10,"max_completion_tokens":0.0`, 10},
		{`"max_tokens":10,"max_completion_tokens":false`, 10},
		{`"max_tokens":10,"max_completion_tokens":null`, 10},
		{`"max_tokens":10,"max_completion_tokens":-1`, 90},
		{`"max_tokens":"ignored","max_completion_tokens":20`, 20},
	} {
		eng.requested = 0
		body := `{"messages":[{"role":"user","content":"hi"}],` + tc.fields + `}`
		response := responsesRequest(handler, "http://127.0.0.1/v1/chat/completions", body, "")
		if response.Code != http.StatusOK || eng.requested != tc.want {
			t.Errorf("%s: status %d, requested %d, want %d: %s", tc.fields, response.Code, eng.requested, tc.want, response.Body)
		}
	}
}
