package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type cacheUsageEngine struct{}

func (cacheUsageEngine) Alive() bool       { return true }
func (cacheUsageEngine) ContextLimit() int { return 4096 }
func (cacheUsageEngine) GenerateWithOptions(_ context.Context, _ []int, _ int, opts engine.RequestOptions) (engine.Result, error) {
	if err := opts.OnToken('A'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop", Extra: []string{"0", "0", "1"}}, nil
}

func TestCacheUsageAcrossAPIs(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: cacheUsageEngine{}, Renderer: contextRenderer{}, MaxBody: 8192,
		MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`},
		{"/v1/responses", `{"input":"hi"}`},
	} {
		w := responsesRequest(h, "http://127.0.0.1"+tc.path, tc.body, "")
		var response struct {
			Usage struct {
				PromptDetails struct {
					Cached int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
				InputDetails struct {
					Cached int `json:"cached_tokens"`
				} `json:"input_tokens_details"`
				CacheRead int `json:"cache_read_input_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || w.Code != http.StatusOK {
			t.Fatalf("%s: status %d, error %v: %s", tc.path, w.Code, err, w.Body.String())
		}
		got := response.Usage.PromptDetails.Cached
		if tc.path == "/v1/messages" {
			got = response.Usage.CacheRead
		} else if tc.path == "/v1/responses" {
			got = response.Usage.InputDetails.Cached
		}
		if got != 1 {
			t.Errorf("%s: cached = %d, want 1: %s", tc.path, got, w.Body.String())
		}
		streamBody := strings.TrimSuffix(tc.body, "}") + `,"stream":true}`
		stream := responsesRequest(h, "http://127.0.0.1"+tc.path, streamBody, "")
		needle := `"cached_tokens":1`
		if tc.path == "/v1/messages" {
			needle = `"cache_read_input_tokens":1`
		}
		if stream.Code != http.StatusOK || !strings.Contains(stream.Body.String(), needle) {
			t.Errorf("%s streaming: status %d: %s", tc.path, stream.Code, stream.Body.String())
		}
	}
}

func TestCachedPromptTokens(t *testing.T) {
	for _, tc := range []struct {
		name    string
		result  engine.Result
		stopped bool
		want    int
		bad     bool
	}{
		{"old engine", engine.Result{Finish: "stop"}, false, 0, false},
		{"reuse", engine.Result{Finish: "stop", Extra: []string{"0", "0", "2"}}, false, 2, false},
		{"read count", engine.Result{Finish: "stop", Extra: []string{"0", "0", "2", "0", "0", "0", "0", "0", "3"}}, false, 7, false},
		{"stopped", engine.Result{Finish: "stop", Extra: []string{"0", "0", "2", "0", "0", "0", "0", "0", "3"}}, true, 2, false},
		{"clamped", engine.Result{Finish: "stop", Extra: []string{"0", "0", "99"}}, false, 10, false},
		{"invalid reuse", engine.Result{Finish: "stop", Extra: []string{"0", "0", "bad"}}, false, 0, true},
		{"negative read", engine.Result{Finish: "stop", Extra: []string{"0", "0", "2", "0", "0", "0", "0", "0", "-1"}}, false, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := cachedPromptTokens(tc.result, 10, tc.stopped)
			if got != tc.want || (err != nil) != tc.bad {
				t.Fatalf("cached = %d, error %v; want %d, bad=%v", got, err, tc.want, tc.bad)
			}
		})
	}
}
