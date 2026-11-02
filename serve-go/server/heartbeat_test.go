package server

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"strata/engine"
	"strata/requestnorm"
)

type heartbeatEngine struct {
	release   <-chan struct{}
	text      string
	cancelled chan<- struct{}
}

func (heartbeatEngine) Alive() bool       { return true }
func (heartbeatEngine) ContextLimit() int { return 4096 }
func (e heartbeatEngine) GenerateWithOptions(ctx context.Context, _ []int, _ int, o engine.RequestOptions) (engine.Result, error) {
	if o.OnProgress != nil {
		if err := o.OnProgress(); err != nil {
			return engine.Result{}, err
		}
	}
	select {
	case <-e.release:
	case <-ctx.Done():
		if e.cancelled != nil {
			close(e.cancelled)
		}
		return engine.Result{}, ctx.Err()
	}
	for _, b := range []byte(e.text) {
		if err := o.OnToken(int(b)); err != nil {
			return engine.Result{}, err
		}
	}
	return engine.Result{Finish: "stop"}, nil
}

func TestStreamPromptProgress(t *testing.T) {
	for _, tc := range []struct{ name, path, body, text string }{
		{"chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}`, "A"},
		{"structured chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true,"response_format":{"type":"json_object"}}`, `{"x":1}`},
		{"anthropic", "/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}`, "A"},
		{"responses", "/v1/responses", `{"input":"hi","max_output_tokens":32,"stream":true}`, "A"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tok := sharedTokenizer(t)
			release := make(chan struct{})
			h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
				Engine: heartbeatEngine{release: release, text: tc.text}, Renderer: contextRenderer{}, MaxBody: 8192,
				MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 1})
			if err != nil {
				t.Fatal(err)
			}
			ts := httptest.NewServer(h)
			defer ts.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+tc.path, strings.NewReader(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			resp, err := ts.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			reader := bufio.NewReader(resp.Body)
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					close(release)
					t.Fatal(err)
				}
				if line == ": keep-alive\n" {
					break
				}
			}
			close(release)
			rest, err := io.ReadAll(reader)
			expected := "A"
			if tc.name == "structured chat" {
				expected = `"finish_reason":"stop"`
			}
			if err != nil || resp.StatusCode != http.StatusOK || !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") || !strings.Contains(string(rest), expected) {
				t.Fatalf("status=%d content-type=%q rest=%s error=%v", resp.StatusCode, resp.Header.Get("Content-Type"), rest, err)
			}
		})
	}
}

func TestStreamProgressCancellation(t *testing.T) {
	tok := sharedTokenizer(t)
	cancelled := make(chan struct{})
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: heartbeatEngine{release: make(chan struct{}), cancelled: cancelled}, Renderer: contextRenderer{},
		MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	defer ts.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			resp.Body.Close()
			t.Fatal(err)
		}
		if line == ": keep-alive\n" {
			break
		}
	}
	cancel()
	resp.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("engine did not receive cancellation after progress")
	}
}

func TestNonStreamHasNoProgressCallback(t *testing.T) {
	if streamHeartbeat(httptest.NewRecorder(), false, nil) != nil {
		t.Fatal("non-streaming request received a progress callback")
	}
}
