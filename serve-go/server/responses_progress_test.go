package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"strata/engine"
	"strata/requestnorm"
)

type responsesProgressEngine struct{}

func (responsesProgressEngine) Alive() bool       { return true }
func (responsesProgressEngine) ContextLimit() int { return 4096 }
func (responsesProgressEngine) GenerateWithOptions(ctx context.Context, _ []int, _ int, o engine.RequestOptions) (engine.Result, error) {
	if err := o.OnProgress(); err != nil {
		return engine.Result{}, err
	}
	timer := time.NewTimer(16 * time.Second)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return engine.Result{}, ctx.Err()
	case <-timer.C:
	}
	if err := o.OnProgress(); err != nil {
		return engine.Result{}, err
	}
	if err := o.OnToken('A'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop"}, nil
}

func TestResponsesProgressDuringLongWait(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: responsesProgressEngine{}, Renderer: responsesRenderer{}, MaxBody: 8192,
		MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	defer ts.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/responses",
		strings.NewReader(`{"input":"hi","max_output_tokens":32,"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	progress, seq, completed := 0, 0, false
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var event struct {
			Type string `json:"type"`
			Seq  int    `json:"sequence_number"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatal(err)
		}
		if event.Seq != seq {
			t.Fatalf("event %s sequence=%d want=%d", event.Type, event.Seq, seq)
		}
		seq++
		if event.Type == "response.in_progress" {
			progress++
		}
		if event.Type == "response.completed" {
			completed = true
		}
	}
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(raw), ": keep-alive\n\n") || progress != 2 || !completed {
		t.Fatalf("status=%d progress=%d completed=%t body=%s", resp.StatusCode, progress, completed, raw)
	}
}
