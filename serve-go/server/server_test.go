package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"strata/engine"
	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
)

// The test binary provides a native engine fixture over real pipes.
func TestHelperProcess(t *testing.T) {
	if len(os.Args) < 4 || os.Args[2] != "--" {
		return
	}
	switch os.Args[3] {
	case "engine":
		fmt.Println("READY 4096 stop")
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			if scanner.Text() == "QUIT" {
				os.Exit(0)
			}
			if strings.HasPrefix(scanner.Text(), "GEN ") {
				for _, b := range []byte("hello STOPhidden") {
					fmt.Printf("T %d\n", b)
				}
				fmt.Println("DONE 16 2 1 1 stop")
			}
		}
		os.Exit(0)
	}
}

func syntheticTokenizer(t *testing.T) (*modeltokenizer.Tokenizer, string) {
	t.Helper()
	dir := t.TempDir()
	vocab := map[string]int{}
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = rune(256 + n)
			n++
		}
		vocab[string(r)] = b
	}
	write := func(name string, v any) {
		t.Helper()
		b, e := json.Marshal(v)
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(filepath.Join(dir, name), b, 0600); e != nil {
			t.Fatal(e)
		}
	}
	write("tokenizer.json", map[string]string{"pre": "qwen35", "model": "gpt2"})
	write("vocab.json", vocab)
	write("token_type.json", make([]int, 256))
	if err := os.WriteFile(filepath.Join(dir, "merges.txt"), []byte("a b"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(`{{ messages[0].content }}`), 0600); err != nil {
		t.Fatal(err)
	}
	tok, err := modeltokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return tok, dir
}

func setup(t *testing.T) (http.Handler, *engine.Engine) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tok, dir := syntheticTokenizer(t)
	renderer, err := NewTemplateRenderer(dir)
	if err != nil {
		t.Fatal(err)
	}
	eng, err := engine.Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "engine"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = eng.Close(ctx)
	})
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: eng, Renderer: renderer, APIKeys: []string{"secret"}, MaxBody: 4096, MaxPromptBytes: 2048, MaxOutputTokens: 100, MaxConcurrent: 2})
	if err != nil {
		t.Fatal(err)
	}
	return h, eng
}

func request(t *testing.T, client *http.Client, url, body, key string) (int, string) {
	t.Helper()
	req, err := http.NewRequest("POST", url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, string(b)
}

func TestHTTPIntegration(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	url := ts.URL + "/v1/chat/completions"
	body := `{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}`
	status, bodyOut := request(t, ts.Client(), url, body, "secret")
	if status != 200 {
		t.Fatalf("status %d: %s", status, bodyOut)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Finish string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(bodyOut), &out); err != nil || len(out.Choices) != 1 || out.Choices[0].Message.Content != "hello STOPhidden" || out.Choices[0].Finish != "stop" {
		t.Fatalf("response %s: %v", bodyOut, err)
	}
	status, bodyOut = request(t, ts.Client(), url, `{"model":"test","stream":true,"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}`, "secret")
	if status != 200 || !strings.Contains(bodyOut, `"content":"h"`) || !strings.Contains(bodyOut, `"finish_reason":"stop"`) || !strings.Contains(bodyOut, "data: [DONE]") {
		t.Fatalf("stream %d %s", status, bodyOut)
	}
	status, _ = request(t, ts.Client(), url, body, "")
	if status != 401 {
		t.Fatalf("unauthorized status %d", status)
	}
	status, _ = request(t, ts.Client(), url, `{"messages":[{"role":"user","content":"Hi"}],"n":2}`, "secret")
	if status != 400 {
		t.Fatalf("unsupported field status %d", status)
	}
	status, _ = request(t, ts.Client(), ts.URL+"/v1/messages", `{}`, "secret")
	if status != 501 {
		t.Fatalf("unsupported endpoint status %d", status)
	}
	status, bodyOut = request(t, ts.Client(), url, `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stop":"STOP","chat_template_kwargs":{"enable_thinking":false}}`, "secret")
	if status != 200 || !strings.Contains(bodyOut, `"content":"hello "`) {
		t.Fatalf("stop response %d %s", status, bodyOut)
	}
	status, bodyOut = request(t, ts.Client(), url, body, "secret")
	if status != 200 || !strings.Contains(bodyOut, `"content":"hello STOPhidden"`) {
		t.Fatalf("request after stop %d %s", status, bodyOut)
	}
}

func TestRendererRequired(t *testing.T) {
	if _, err := NewTemplateRenderer(t.TempDir()); err == nil {
		t.Fatal("missing renderer accepted")
	}
}
