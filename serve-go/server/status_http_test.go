package server

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type statusEngine struct{}

type statusOutput struct {
	bytes.Buffer
	writes int
}

func (w *statusOutput) Write(p []byte) (int, error) {
	w.writes++
	return w.Buffer.Write(p)
}

func (statusEngine) Alive() bool       { return true }
func (statusEngine) ContextLimit() int { return 4096 }
func (statusEngine) GenerateWithOptions(_ context.Context, _ []int, _ int, opts engine.RequestOptions) (engine.Result, error) {
	if opts.OnPromptProgress != nil {
		opts.OnPromptProgress(1, 2)
	}
	if err := opts.OnToken('A'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Generated: 1, Prompt: 2, DecodeMS: 1, Finish: "stop"}, nil
}

func TestStatusLoggingPreservesAPIResponses(t *testing.T) {
	const secret = "private-client-prompt"
	for _, tc := range []struct {
		path string
		body string
	}{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"` + secret + `"}],"max_tokens":32}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"` + secret + `"}],"max_tokens":32}`},
		{"/v1/responses", `{"input":"` + secret + `","max_output_tokens":32}`},
	} {
		t.Run(tc.path, func(t *testing.T) {
			tok := sharedTokenizer(t)
			var output statusOutput
			h, err := New(Config{
				Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
				Engine: statusEngine{}, Renderer: contextRenderer{},
				MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 1,
				StatusLogger: log.New(&output, "", 0),
			})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1"+tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			h.ServeHTTP(recorder, req)
			var body any
			if recorder.Code != http.StatusOK || json.Unmarshal(recorder.Body.Bytes(), &body) != nil ||
				!strings.Contains(recorder.Body.String(), `"A"`) {
				t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			if output.writes < 2 || strings.Contains(output.String(), secret) {
				t.Fatal("status output missing or contains private request content")
			}
		})
	}
}
