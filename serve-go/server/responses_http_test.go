package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
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
	tok := sharedTokenizer(t)
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
			Input   int `json:"input_tokens"`
			Output  int `json:"output_tokens"`
			Details struct {
				Reasoning int `json:"reasoning_tokens"`
			} `json:"output_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Status != "completed" || len(response.Output) != 2 || response.Output[0].Type != "reasoning" ||
		response.Output[0].Content[0].Text != "thinking" || response.Output[1].Content[0].Text != "answer" ||
		response.Usage.Input != 2 || response.Usage.Output != len("thinking</think>answer") ||
		response.Usage.Details.Reasoning != len("thinking</think>") {
		t.Fatalf("unexpected response: %s", w.Body.String())
	}
}

func TestRepeatStopAcrossAPIs(t *testing.T) {
	tok := sharedTokenizer(t)
	for _, tc := range []struct {
		limit      int
		wantText   string
		wantFinish string
		wantTokens int
	}{
		{3, "aa", "length", 3},
		{0, "aaaab", "stop", 5},
	} {
		h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
			Engine: responsesEngine{text: "aaaab", finish: "stop"}, Renderer: contextRenderer{},
			APIKeys: []string{"secret"}, MaxBody: 8192, MaxPromptBytes: 2048,
			MaxOutputTokens: 512, MaxConcurrent: 1, RepeatStopTokens: &tc.limit})
		if err != nil {
			t.Fatal(err)
		}
		for _, api := range []struct {
			path, body, text, finish, tokens string
		}{
			{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, `"content":"` + tc.wantText + `"`, `"finish_reason":"` + tc.wantFinish + `"`, `"completion_tokens":`},
			{"/v1/messages", `{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":32}`, `"text":"` + tc.wantText + `"`, `"stop_reason":"` + map[string]string{"length": "max_tokens", "stop": "end_turn"}[tc.wantFinish] + `"`, `"output_tokens":`},
			{"/v1/responses", `{"input":"hi"}`, `"text":"` + tc.wantText + `"`, `"status":"` + map[string]string{"length": "incomplete", "stop": "completed"}[tc.wantFinish] + `"`, `"output_tokens":`},
		} {
			w := responsesRequest(h, api.path, api.body, "secret")
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), api.text) ||
				!strings.Contains(w.Body.String(), api.finish) ||
				!strings.Contains(w.Body.String(), api.tokens+strconv.Itoa(tc.wantTokens)) {
				t.Errorf("limit %d %s: status %d: %s", tc.limit, api.path, w.Code, w.Body.String())
			}
		}
	}
}

type samplingCapture struct {
	mu  sync.Mutex
	got engine.Sampling
}

func (*samplingCapture) Alive() bool       { return true }
func (*samplingCapture) ContextLimit() int { return 4096 }
func (e *samplingCapture) GenerateWithOptions(_ context.Context, _ []int, _ int, opts engine.RequestOptions) (engine.Result, error) {
	e.mu.Lock()
	e.got = opts.Sampling
	e.mu.Unlock()
	if err := opts.OnToken('x'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop"}, nil
}

func TestResponsesNumericStringSampling(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &samplingCapture{}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: responsesRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"input":"Hi","max_output_tokens":32,"temperature":"0.7","top_p":"0.9"}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/responses", body, "secret")
	var response struct {
		Temperature float64 `json:"temperature"`
		TopP        float64 `json:"top_p"`
	}
	err = json.Unmarshal([]byte(raw), &response)
	eng.mu.Lock()
	defer eng.mu.Unlock()
	if status != http.StatusOK || err != nil || response.Temperature != 0.7 || response.TopP != 0.9 ||
		eng.got.Temperature == nil || *eng.got.Temperature != 0.7 || eng.got.TopP == nil || *eng.got.TopP != 0.9 {
		t.Fatalf("status %d, response %s, error %v", status, raw, err)
	}
}

func TestResponsesCustomToolCalls(t *testing.T) {
	for _, tc := range []struct {
		tool      string
		modelName string
		namespace string
	}{
		{`{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: \"x\""}}`, "apply_patch", ""},
		{`{"type":"namespace","name":"fs","tools":[{"type":"custom","name":"apply_patch"}]}`, "fs.apply_patch", "fs"},
	} {
		text := `<tool_call><function=` + tc.modelName + `><parameter=input>echo hi</parameter></function></tool_call>`
		for _, stream := range []string{"false", "true"} {
			h := responsesHandler(t, text, "stop", false)
			body := `{"input":"hello","stream":` + stream + `,"tools":[` + tc.tool + `]}`
			w := responsesRequest(h, "/v1/responses", body, "secret")
			if w.Code != http.StatusOK {
				t.Fatalf("%s stream=%s: status %d: %s", tc.modelName, stream, w.Code, w.Body.String())
			}
			var response struct {
				Output []struct {
					Type      string `json:"type"`
					Name      string `json:"name"`
					Namespace string `json:"namespace"`
					Input     string `json:"input"`
				} `json:"output"`
			}
			if stream == "false" {
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
			} else {
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if strings.HasPrefix(line, "data: ") {
						var event struct {
							Type     string          `json:"type"`
							Response json.RawMessage `json:"response"`
						}
						if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
							t.Fatal(err)
						}
						if event.Type == "response.completed" {
							if err := json.Unmarshal(event.Response, &response); err != nil {
								t.Fatal(err)
							}
						}
					}
				}
				if !strings.Contains(w.Body.String(), "event: response.custom_tool_call_input.delta\n") ||
					!strings.Contains(w.Body.String(), "event: response.custom_tool_call_input.done\n") {
					t.Fatalf("missing custom tool stream events: %s", w.Body.String())
				}
			}
			if len(response.Output) != 1 || response.Output[0].Type != "custom_tool_call" ||
				response.Output[0].Name != "apply_patch" || response.Output[0].Namespace != tc.namespace ||
				response.Output[0].Input != "echo hi" {
				t.Fatalf("%s stream=%s: unexpected response: %s", tc.modelName, stream, w.Body.String())
			}
		}
	}
}

func TestResponsesHostedToolsAreLeftOut(t *testing.T) {
	tok := sharedTokenizer(t)
	for _, tc := range []struct {
		tools      string
		text       string
		wantTools  int
		outputType string
	}{
		{`[{"type":"web_search"},{"type":"future_tool"}]`, "</think>hello", 0, "message"},
		{`[{"type":"web_search"},{"type":"function","name":"lookup","parameters":{"type":"object"}}]`,
			"</think><tool_call><function=lookup></function></tool_call>", 1, "function_call"},
	} {
		for _, stream := range []string{"false", "true"} {
			renderer := &toolChoiceRenderer{}
			h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
				Engine: responsesEngine{text: tc.text, finish: "stop"}, Renderer: renderer, APIKeys: []string{"secret"},
				MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
			if err != nil {
				t.Fatal(err)
			}
			body := `{"input":"hi","tools":` + tc.tools + `,"stream":` + stream + `}`
			w := responsesRequest(h, "/v1/responses", body, "secret")
			renderer.mu.Lock()
			gotTools := renderer.tools
			renderer.mu.Unlock()
			if w.Code != http.StatusOK || gotTools != tc.wantTools ||
				!strings.Contains(w.Body.String(), `"type":"`+tc.outputType+`"`) {
				t.Errorf("tools=%s stream=%s: status %d, offered %d: %s", tc.tools, stream, w.Code, gotTools, w.Body.String())
			}
		}
	}
}

func TestResponsesNumericStringSamplingValidation(t *testing.T) {
	h := responsesHandler(t, "answer", "stop", false)
	for _, tc := range []struct {
		field string
		want  int
	}{
		{`"temperature":" 0.7 "`, http.StatusOK},
		{`"top_p":"1"`, http.StatusOK},
		{`"temperature":"not-a-number"`, http.StatusBadRequest},
		{`"temperature":"NaN"`, http.StatusBadRequest},
		{`"top_p":"Infinity"`, http.StatusBadRequest},
		{`"top_p":true`, http.StatusBadRequest},
		{`"top_p":"0"`, http.StatusBadRequest},
	} {
		body := `{"input":"hi",` + tc.field + `}`
		w := responsesRequest(h, "/v1/responses", body, "secret")
		if w.Code != tc.want {
			t.Errorf("%s: status %d, want %d: %s", tc.field, w.Code, tc.want, w.Body.String())
		}
	}
}

func TestResponsesAdditionalSampling(t *testing.T) {
	tok := sharedTokenizer(t)
	eng := &samplingCapture{}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: responsesRenderer{}, APIKeys: []string{"secret"}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, stream := range []string{"false", "true"} {
		body := `{"input":"Hi","stream":` + stream + `,"top_k":"20","min_p":"0.1",` +
			`"repeat_penalty":2,"repetition_penalty":1.1,"frequency_penalty":-0.5,` +
			`"presence_penalty":0.2,"repeat_last_n":10,"penalty_last_n":"16","seed":"123"}`
		w := responsesRequest(h, "/v1/responses", body, "secret")
		if w.Code != http.StatusOK {
			t.Fatalf("stream=%s: status %d: %s", stream, w.Code, w.Body.String())
		}
		eng.mu.Lock()
		s := eng.got
		eng.mu.Unlock()
		if s.TopK == nil || *s.TopK != 20 || s.MinP == nil || *s.MinP != 0.1 ||
			s.RepetitionPenalty == nil || *s.RepetitionPenalty != 1.1 ||
			s.FrequencyPenalty == nil || *s.FrequencyPenalty != -0.5 ||
			s.PresencePenalty == nil || *s.PresencePenalty != 0.2 ||
			s.PenaltyLastN == nil || *s.PenaltyLastN != 16 || s.Seed == nil || *s.Seed != 123 {
			t.Fatalf("stream=%s: unexpected sampling %#v", stream, s)
		}
	}
	for _, tc := range []struct {
		value string
		want  int
	}{
		{"0", 64},
		{"100", 64},
		{"-1", 20},
		{"null", 20},
		{`"4.0"`, 4},
	} {
		w := responsesRequest(h, "/v1/responses", `{"input":"Hi","top_k":`+tc.value+`}`, "secret")
		eng.mu.Lock()
		got := eng.got.TopK
		eng.mu.Unlock()
		if w.Code != http.StatusOK || got == nil || *got != tc.want {
			t.Errorf("top_k=%s: status %d, top_k %v: %s", tc.value, w.Code, got, w.Body.String())
		}
	}
}

func TestResponsesAdditionalSamplingValidation(t *testing.T) {
	h := responsesHandler(t, "answer", "stop", false)
	for _, tc := range []struct {
		field string
		want  int
	}{
		{`"top_k":0`, http.StatusOK},
		{`"top_k":100`, http.StatusOK},
		{`"top_k":-1`, http.StatusOK},
		{`"top_k":"1.5"`, http.StatusBadRequest},
		{`"top_k":true`, http.StatusBadRequest},
		{`"min_p":"NaN"`, http.StatusBadRequest},
		{`"seed":true`, http.StatusBadRequest},
		{`"penalty_last_n":1.5`, http.StatusBadRequest},
	} {
		w := responsesRequest(h, "/v1/responses", `{"input":"hi",`+tc.field+`}`, "secret")
		if w.Code != tc.want {
			t.Errorf("%s: status %d, want %d: %s", tc.field, w.Code, tc.want, w.Body.String())
		}
	}
}

func TestResponsesReasoningTokenUsageStream(t *testing.T) {
	for _, tc := range []struct {
		thinking bool
		text     string
		want     int
	}{
		{true, "abc</think>answer", len("abc</think>")},
		{false, "answer", 0},
		{true, "unfinished", len("unfinished")},
	} {
		h := responsesHandler(t, tc.text, "stop", tc.thinking)
		w := responsesRequest(h, "/v1/responses", `{"input":"hello","stream":true}`, "secret")
		found := false
		for _, line := range strings.Split(w.Body.String(), "\n") {
			if !strings.HasPrefix(line, "data: {") {
				continue
			}
			var event struct {
				Type     string `json:"type"`
				Response struct {
					Usage struct {
						Details struct {
							Reasoning int `json:"reasoning_tokens"`
						} `json:"output_tokens_details"`
					} `json:"usage"`
				} `json:"response"`
			}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
				t.Fatal(err)
			}
			if event.Type == "response.completed" {
				found = true
				if event.Response.Usage.Details.Reasoning != tc.want {
					t.Errorf("%q: reasoning tokens %d, want %d", tc.text, event.Response.Usage.Details.Reasoning, tc.want)
				}
			}
		}
		if w.Code != http.StatusOK || !found {
			t.Errorf("%q: status %d: %s", tc.text, w.Code, w.Body.String())
		}
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

func TestResponsesStopSequences(t *testing.T) {
	h := responsesHandler(t, "before STOP hidden", "stop", false)
	for _, tc := range []struct {
		fields string
		want   string
	}{
		{`"stop":"STOP"`, "before "},
		{`"stop":["STOP","hidden"]`, "before "},
		{`"stop":null,"stop_sequences":["STOP"]`, "before "},
		{`"stop":"hidden","stop_sequences":["STOP"]`, "before STOP "},
		{`"stop":[],"stop_sequences":["STOP"]`, "before STOP hidden"},
	} {
		for _, stream := range []bool{false, true} {
			body := `{"input":"hello",` + tc.fields
			if stream {
				body += `,"stream":true`
			}
			w := responsesRequest(h, "/v1/responses", body+`}`, "secret")
			if w.Code != http.StatusOK {
				t.Fatalf("%s stream=%v: %d %s", tc.fields, stream, w.Code, w.Body.String())
			}
			var text string
			if stream {
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if !strings.HasPrefix(line, "data: ") {
						continue
					}
					var event struct {
						Type  string `json:"type"`
						Delta string `json:"delta"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
						t.Fatal(err)
					}
					if event.Type == "response.output_text.delta" {
						text += event.Delta
					}
				}
				if !strings.Contains(w.Body.String(), "event: response.completed\n") {
					t.Errorf("missing completion: %s", w.Body.String())
				}
			} else {
				var response struct {
					Output []struct {
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"output"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || len(response.Output) != 1 || len(response.Output[0].Content) != 1 {
					t.Fatalf("response %s: %v", w.Body.String(), err)
				}
				text = response.Output[0].Content[0].Text
			}
			if text != tc.want {
				t.Errorf("%s stream=%v: text %q, want %q", tc.fields, stream, text, tc.want)
			}
		}
	}
}

func TestResponsesInvalidStopSequences(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	for _, fields := range []string{
		`"stop":42`, `"stop":["x",1]`, `"stop":["a","b","c","d","e"]`,
		`"stop_sequences":"x"`, `"stop_sequences":[null]`,
	} {
		w := responsesRequest(h, "/v1/responses", `{"input":"hello",`+fields+`}`, "secret")
		if w.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d: %s", fields, w.Code, w.Body.String())
		}
	}
}

func TestResponsesStopIgnoresReasoningAndToolBoundaries(t *testing.T) {
	t.Run("reasoning", func(t *testing.T) {
		h := responsesHandler(t, "STOP thinking</think>visible STOP hidden", "stop", true)
		w := responsesRequest(h, "/v1/responses", `{"input":"hello","stop":"STOP"}`, "secret")
		var response struct {
			Output []struct {
				Type    string `json:"type"`
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"output"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusOK ||
			len(response.Output) != 2 || response.Output[0].Content[0].Text != "STOP thinking" ||
			response.Output[1].Content[0].Text != "visible " {
			t.Fatalf("status %d: %s, %v", w.Code, w.Body.String(), err)
		}
	})
	t.Run("tool boundary", func(t *testing.T) {
		text := "before ST<tool_call><function=lookup></function></tool_call>OP after"
		h := responsesHandler(t, text, "stop", false)
		body := `{"input":"hello","stop":"STOP","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`
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
			len(response.Output) != 3 || response.Output[0].Content[0].Text != "before ST" ||
			response.Output[1].Type != "function_call" || response.Output[2].Content[0].Text != "OP after" {
			t.Fatalf("status %d: %s, %v", w.Code, w.Body.String(), err)
		}
	})
}

func TestResponsesToolChoicesMatchStatelessPython(t *testing.T) {
	text := "<tool_call><function=lookup><parameter=x>7</parameter></function></tool_call>"
	h := responsesHandler(t, text, "stop", false)
	for _, tc := range []struct {
		choice   string
		wantCall bool
	}{
		{`"none"`, false},
		{`"auto"`, true},
		{`"required"`, true},
		{`{"type":"function","name":"lookup"}`, true},
		{`"unexpected"`, true},
		{`null`, true},
	} {
		body := `{"input":"hello","tool_choice":` + tc.choice + `,"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}]}`
		w := responsesRequest(h, "/v1/responses", body, "secret")
		var response struct {
			ToolChoice json.RawMessage `json:"tool_choice"`
			Output     []struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"output"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusOK {
			t.Errorf("%s: status %d, error %v: %s", tc.choice, w.Code, err, w.Body.String())
			continue
		}
		var got, want any
		if err := json.Unmarshal(response.ToolChoice, &got); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal([]byte(tc.choice), &want); err != nil {
			t.Fatal(err)
		}
		gotJSON, _ := json.Marshal(got)
		wantJSON, _ := json.Marshal(want)
		if string(gotJSON) != string(wantJSON) {
			t.Errorf("choice = %s, want %s", gotJSON, wantJSON)
		}
		hasCall := false
		for _, item := range response.Output {
			hasCall = hasCall || item.Type == "function_call" && item.Name == "lookup"
		}
		if hasCall != tc.wantCall {
			t.Errorf("%s: function call = %v, want %v: %s", tc.choice, hasCall, tc.wantCall, w.Body.String())
		}
	}
}

func TestResponsesRequiredDoesNotForceTool(t *testing.T) {
	h := responsesHandler(t, "plain answer", "stop", false)
	body := `{"input":"hello","tool_choice":"required","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`
	for _, stream := range []bool{false, true} {
		requestBody := body
		if stream {
			requestBody = strings.TrimSuffix(body, "}") + `,"stream":true}`
		}
		w := responsesRequest(h, "/v1/responses", requestBody, "secret")
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "plain answer") ||
			strings.Contains(w.Body.String(), `"type":"function_call"`) {
			t.Errorf("stream=%v: status %d: %s", stream, w.Code, w.Body.String())
		}
	}
}

func TestResponsesErrorsAndChatRegression(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	for _, body := range []string{
		`{"input":"x","stream":"yes"}`,
		`{"input":"x","previous_response_id":"old"}`,
		`{"input":"x","text":{"format":{"type":"json_schema","name":"x","schema":{"type":"array"}}}}`,
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

func TestChatTextResponseFormat(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	for _, format := range []string{`null`, `{"type":"text"}`} {
		body := `{"messages":[{"role":"user","content":"hi"}],"response_format":` + format + `}`
		w := responsesRequest(h, "/v1/chat/completions", body, "secret")
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"content":"hello"`) {
			t.Errorf("%s: %d %s", format, w.Code, w.Body.String())
		}
	}
	w := responsesRequest(h, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema"}}`, "secret")
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unsupported structured format: %d %s", w.Code, w.Body.String())
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
