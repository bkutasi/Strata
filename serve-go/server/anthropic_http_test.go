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

	"strata/requestnorm"
)

func TestAnthropicMessagesHTTP(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"thinking":{"type":"disabled"}}`
	for _, path := range []string{"/v1/messages", "/v1/v1/messages"} {
		status, raw := request(t, ts.Client(), ts.URL+path, body, "secret")
		var response struct {
			Type       string `json:"type"`
			StopReason string `json:"stop_reason"`
			Content    []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal([]byte(raw), &response); status != http.StatusOK || err != nil ||
			response.Type != "message" || response.StopReason != "end_turn" ||
			len(response.Content) != 1 || response.Content[0].Type != "text" || response.Content[0].Text != "hello STOPhidden" {
			t.Fatalf("%s: status %d, body %s, error %v", path, status, raw, err)
		}
	}
	stream := strings.TrimSuffix(body, "}") + `,"stream":true}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/messages", stream, "secret")
	if status != http.StatusOK || !strings.Contains(raw, "event: message_start\n") ||
		!strings.Contains(raw, "event: content_block_delta\n") || !strings.Contains(raw, `"text":"h"`) ||
		!strings.Contains(raw, `"stop_reason":"end_turn"`) || !strings.Contains(raw, "event: message_stop\n") {
		t.Fatalf("stream status %d: %s", status, raw)
	}
	stop := strings.TrimSuffix(body, "}") + `,"stop_sequences":["STOP"]}`
	status, raw = request(t, ts.Client(), ts.URL+"/v1/messages", stop, "secret")
	if status != http.StatusOK || !strings.Contains(raw, `"stop_reason":"stop_sequence"`) ||
		!strings.Contains(raw, `"stop_sequence":"STOP"`) || !strings.Contains(raw, `"text":"hello "`) {
		t.Fatalf("stop status %d: %s", status, raw)
	}
	status, _ = request(t, ts.Client(), ts.URL+"/v1/messages", body, "")
	if status != http.StatusUnauthorized {
		t.Fatalf("unauthorized status %d", status)
	}
	status, _ = request(t, ts.Client(), ts.URL+"/v1/messages", `{"messages":[]}`, "secret")
	if status != http.StatusBadRequest {
		t.Fatalf("empty messages status %d", status)
	}
}

func TestAnthropicDeepSeekReasoningFallbackHTTP(t *testing.T) {
	h, _ := setupFamilyEngine(t, "engine-deep-reason", requestnorm.FamilyDeepSeek)
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/messages", body, "secret")
	var response struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &response); status != http.StatusOK || err != nil ||
		len(response.Content) != 1 || response.Content[0].Type != "text" || response.Content[0].Text != "fallback answer" {
		t.Fatalf("status %d, body %s, error %v", status, raw, err)
	}
}

func TestAnthropicStreamingBeforeEngineFinishes(t *testing.T) {
	h, _ := setupEngine(t, "engine-cancel")
	ts := httptest.NewServer(h)
	defer ts.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":true,"thinking":{"type":"disabled"}}`
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/messages", strings.NewReader(body))
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
		if strings.Contains(scanner.Text(), `"text":"h"`) {
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

func TestAnthropicDeepSeekToolHTTP(t *testing.T) {
	h, _ := setupFamilyEngine(t, "engine-deep-tool", requestnorm.FamilyDeepSeek)
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":200,"thinking":{"type":"disabled"},` +
		`"tools":[{"name":"lookup","input_schema":{"type":"object","properties":{"x":{"type":"integer"}}}}]}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/messages", body, "secret")
	var response struct {
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type  string         `json:"type"`
			Name  string         `json:"name"`
			Input map[string]any `json:"input"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &response); status != http.StatusOK || err != nil ||
		response.StopReason != "tool_use" || len(response.Content) != 1 ||
		response.Content[0].Type != "tool_use" || response.Content[0].Name != "lookup" ||
		response.Content[0].Input["x"] != "123" {
		t.Fatalf("status %d, body %s, error %v", status, raw, err)
	}
}
