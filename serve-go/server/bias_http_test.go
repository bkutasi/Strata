package server

import (
	"context"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type biasCapture struct {
	supported bool
	calls     int
	sampling  engine.Sampling
}

func (*biasCapture) Alive() bool               { return true }
func (*biasCapture) ContextLimit() int         { return 4096 }
func (e *biasCapture) SupportsLogitBias() bool { return e.supported }
func (e *biasCapture) GenerateWithOptions(_ context.Context, _ []int, _ int, opts engine.RequestOptions) (engine.Result, error) {
	e.calls++
	e.sampling = opts.Sampling
	if err := opts.OnToken('A'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop"}, nil
}

func TestChatLogitBiasHTTP(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	eng := &biasCapture{supported: true}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 128, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, stream := range []string{"false", "true"} {
		body := `{"messages":[{"role":"user","content":"hi"}],"logit_bias":[[9,2.5],[7,false]],"stream":` + stream + `}`
		w := responsesRequest(h, "http://127.0.0.1/v1/chat/completions", body, "")
		if w.Code != http.StatusOK || !reflect.DeepEqual(eng.sampling.LogitBias, map[int]float64{7: -100, 9: 2.5}) {
			t.Fatalf("stream=%s status=%d bias=%v body=%s", stream, w.Code, eng.sampling.LogitBias, w.Body)
		}
	}
	for _, bias := range []string{`{"260":0}`, `[[1,true]]`, `[[1,0],["01",1]]`} {
		before := eng.calls
		w := responsesRequest(h, "http://127.0.0.1/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"stream":true,"logit_bias":`+bias+`}`, "")
		if w.Code != http.StatusBadRequest || eng.calls != before || !strings.Contains(w.Body.String(), "logit_bias") {
			t.Errorf("bias %s: status=%d calls=%d body=%s", bias, w.Code, eng.calls-before, w.Body)
		}
	}
	eng.supported = false
	before := eng.calls
	w := responsesRequest(h, "http://127.0.0.1/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"stream":true,"logit_bias":{"1":-100}}`, "")
	if w.Code != http.StatusBadRequest || eng.calls != before || !strings.Contains(w.Body.String(), "logit_bias") {
		t.Fatalf("unsupported: status=%d calls=%d body=%s", w.Code, eng.calls-before, w.Body)
	}
	w = responsesRequest(h, "http://127.0.0.1/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"logit_bias":{}}`, "")
	if w.Code != http.StatusOK {
		t.Fatalf("empty bias on unsupported engine: %d %s", w.Code, w.Body)
	}
}

func TestOtherAPIsLogitBiasHTTP(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	eng := &biasCapture{supported: true}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 128, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, prefix string }{
		{"/v1/messages", `"messages":[{"role":"user","content":"hi"}],"max_tokens":32`},
		{"/v1/responses", `"input":"hi","max_output_tokens":32`},
	} {
		for _, stream := range []string{"false", "true"} {
			eng.supported = true
			body := `{` + tc.prefix + `,"logit_bias":[[9,2.5],[7,false]],"stream":` + stream + `}`
			w := responsesRequest(h, "http://127.0.0.1"+tc.path, body, "")
			if w.Code != http.StatusOK || !reflect.DeepEqual(eng.sampling.LogitBias, map[int]float64{7: -100, 9: 2.5}) {
				t.Fatalf("%s stream=%s status=%d bias=%v body=%s", tc.path, stream, w.Code, eng.sampling.LogitBias, w.Body)
			}
			before := eng.calls
			eng.supported = false
			w = responsesRequest(h, "http://127.0.0.1"+tc.path, body, "")
			if w.Code != http.StatusBadRequest || eng.calls != before {
				t.Fatalf("%s unsupported: status=%d calls=%d body=%s", tc.path, w.Code, eng.calls-before, w.Body)
			}
			eng.supported = true
			body = `{` + tc.prefix + `,"logit_bias":{"260":1},"stream":` + stream + `}`
			w = responsesRequest(h, "http://127.0.0.1"+tc.path, body, "")
			if w.Code != http.StatusBadRequest || eng.calls != before {
				t.Fatalf("%s invalid: status=%d calls=%d body=%s", tc.path, w.Code, eng.calls-before, w.Body)
			}
		}
	}
	before := eng.calls
	w := responsesRequest(h, "http://127.0.0.1/v1/messages/count_tokens", `{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"bad":"ignored"}}`, "")
	if w.Code != http.StatusOK || eng.calls != before {
		t.Fatalf("count: status=%d calls=%d body=%s", w.Code, eng.calls-before, w.Body)
	}
}
