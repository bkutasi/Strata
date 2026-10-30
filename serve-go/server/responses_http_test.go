package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"strata/engine"
	"strata/requestnorm"
)

type responsesEngine struct {
	text   string
	finish string
}

func (e responsesEngine) Alive() bool       { return true }
func (e responsesEngine) ContextLimit() int { return 4096 }
func (e responsesEngine) GenerateWithOptions(ctx context.Context, _ []int, _ int, o engine.RequestOptions) (engine.Result, error) {
	for _, b := range []byte(e.text) {
		if err := ctx.Err(); err != nil {
			return engine.Result{}, err
		}
		if err := o.OnToken(int(b)); err != nil {
			return engine.Result{}, err
		}
	}
	return engine.Result{Finish: e.finish}, nil
}

type responsesRenderer struct{ thinking bool }

func (r responsesRenderer) Render(context.Context, requestnorm.Result) (Rendered, error) {
	return Rendered{Prompt: "Hi", Thinking: r.thinking}, nil
}

func responsesHandler(t *testing.T, text, finish string, thinking bool) http.Handler {
	t.Helper()
	tok, _ := syntheticTokenizer(t)
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: responsesEngine{text: text, finish: finish}, Renderer: responsesRenderer{thinking: thinking},
		APIKeys: []string{"secret"}, MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 2})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func responsesRequest(h http.Handler, path, body, key string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestResponsesTextAndReasoning(t *testing.T) {
	h := responsesHandler(t, "thinking</think>answer", "stop", true)
	w := responsesRequest(h, "/v1/responses", `{"model":"test","input":"hello"}`, "secret")
	if w.Code != 200 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	var response struct {
		Status string `json:"status"`
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			Input  int `json:"input_tokens"`
			Output int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Status != "completed" || len(response.Output) != 2 || response.Output[0].Type != "reasoning" ||
		response.Output[0].Content[0].Text != "thinking" || response.Output[1].Content[0].Text != "answer" ||
		response.Usage.Input != 2 || response.Usage.Output != len("thinking</think>answer") {
		t.Fatalf("unexpected response: %s", w.Body.String())
	}
}

func TestResponsesToolAndStream(t *testing.T) {
	text := "<tool_call><function=lookup><parameter=x>9007199254740993</parameter><parameter=y>0.1234567890123456789</parameter></function></tool_call>"
	h := responsesHandler(t, text, "stop", false)
	body := `{"input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"number"}}}}]}`
	w := responsesRequest(h, "/v1/responses", body, "secret")
	if w.Code != 200 || !strings.Contains(w.Body.String(), `9007199254740993`) || !strings.Contains(w.Body.String(), `0.12345678901234568`) || !strings.Contains(w.Body.String(), `"status":"completed"`) {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	w = responsesRequest(h, "/v1/responses", strings.TrimSuffix(body, "}")+`,"stream":true}`, "secret")
	for _, event := range []string{"response.created", "response.in_progress", "response.output_item.added", "response.function_call_arguments.delta", "response.output_item.done", "response.completed"} {
		if !strings.Contains(w.Body.String(), "event: "+event+"\n") {
			t.Fatalf("missing %s: %s", event, w.Body.String())
		}
	}
	if !strings.Contains(w.Body.String(), `"usage":`) || !strings.Contains(w.Body.String(), `9007199254740993`) {
		t.Fatal(w.Body.String())
	}
}

func TestResponsesErrorsAndChatRegression(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	for _, body := range []string{
		`{"input":"x","model":"other"}`, `{"input":"x","stream":"yes"}`,
		`{"input":"x","previous_response_id":"old"}`,
		`{"input":"x","text":{"format":{"type":"json_object"}}}`,
		`{"input":[{"role":"user","content":[{"type":"input_image","image_url":"http://example.test"}]}]}`,
		`{"input":"x","max_output_tokens":9999}`, `{"input":"x","top_p":0}`,
	} {
		w := responsesRequest(h, "/v1/responses", body, "secret")
		if w.Code != 400 {
			t.Errorf("%s: %d %s", body, w.Code, w.Body.String())
		}
	}
	if w := responsesRequest(h, "/v1/responses", `{"input":"x"}`, ""); w.Code != 401 {
		t.Fatalf("auth: %d %s", w.Code, w.Body.String())
	}
	w := responsesRequest(h, "/v1/chat/completions", `{"model":"test","messages":[{"role":"user","content":"x"}]}`, "secret")
	if w.Code != 200 || !strings.Contains(w.Body.String(), "hello") {
		t.Fatalf("chat: %d %s", w.Code, w.Body.String())
	}
}

func TestResponsesStoreTrueRemainsStateless(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	w := responsesRequest(h, "/v1/responses", `{"input":"hello","store":true}`, "secret")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"store":false`) {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestResponsesErrorIncludesParameterAndCode(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	w := responsesRequest(h, "/v1/responses", `{"input":"hello","previous_response_id":"old"}`, "secret")
	var body struct {
		Error struct {
			Param string `json:"param"`
			Code  string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil || w.Code != http.StatusBadRequest ||
		body.Error.Param != "previous_response_id" || body.Error.Code != "unsupported_parameter" {
		t.Fatalf("status %d, error %v, body %s", w.Code, err, w.Body.String())
	}
}

func TestChatAcceptsClientMetadata(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	body := `{"messages":[{"role":"user","content":"hi"}],"user":"client","metadata":{"key":"value"},"service_tier":"auto","safety_identifier":"id","prompt_cache_key":"cache"}`
	w := responsesRequest(h, "/v1/chat/completions", body, "secret")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"content":"hello"`) {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestResponsesParallelToolCallsFalse(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	for _, stream := range []string{"false", "true"} {
		body := `{"input":"hello","parallel_tool_calls":false,"stream":` + stream + `}`
		w := responsesRequest(h, "/v1/responses", body, "secret")
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"parallel_tool_calls":false`) {
			t.Errorf("stream=%s: %d %s", stream, w.Code, w.Body.String())
		}
	}
}

func TestResponsesAcceptsCodexMetadata(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	body := `{"input":"hello","prompt_cache_key":"key","client_metadata":{"x-codex-turn-metadata":"{\"request_kind\":\"turn\",\"session_id\":\"s\",\"thread_id\":\"t\"}"}}`
	w := responsesRequest(h, "/v1/responses", body, "secret")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"text":"hello"`) {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
}

func TestResponsesStreamLengthAndIncompleteTool(t *testing.T) {
	h := responsesHandler(t, "partial", "length", false)
	w := responsesRequest(h, "/v1/responses", `{"input":"x","stream":true}`, "secret")
	for _, event := range []string{"response.output_text.delta", "response.incomplete"} {
		if !strings.Contains(w.Body.String(), "event: "+event+"\n") {
			t.Fatalf("missing %s: %s", event, w.Body.String())
		}
	}
	if !strings.Contains(w.Body.String(), `"reason":"max_output_tokens"`) {
		t.Fatal(w.Body.String())
	}
	h = responsesHandler(t, "<tool_call><function=lookup><parameter=x>7", "length", false)
	w = responsesRequest(h, "/v1/responses", `{"input":"x","stream":true,"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}]}`, "secret")
	if strings.Contains(w.Body.String(), `"type":"function_call"`) || !strings.Contains(w.Body.String(), "event: response.incomplete\n") {
		t.Fatal(w.Body.String())
	}
}

func TestResponsesStreamEmitsBeforeEngineFinishes(t *testing.T) {
	h, _ := setupEngine(t, "engine-cancel")
	ts := httptest.NewServer(h)
	defer ts.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	body := `{"input":"Hi","max_output_tokens":32,"stream":true,"reasoning":{"effort":"none"}}`
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/responses", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer secret")
	client := ts.Client()
	client.Timeout = 5 * time.Second
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(resp.Body)
	seen := false
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), `"delta":"h"`) {
			seen = true
			break
		}
	}
	cancel()
	resp.Body.Close()
	if !seen {
		t.Fatalf("no text delta before engine finished: %v", scanner.Err())
	}
}
