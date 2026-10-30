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
	case "engine", "engine-tool", "engine-cancel", "engine-deep-tool", "engine-deep-reason",
		"engine-topk-20", "engine-topk-64", "engine-topk-default":
		fmt.Println("READY 4096 stop")
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			if scanner.Text() == "QUIT" {
				os.Exit(0)
			}
			if strings.HasPrefix(scanner.Text(), "GEN ") {
				if os.Args[3] == "engine-cancel" {
					fmt.Println("T 104")
					if scanner.Scan() && scanner.Text() == "STOP" {
						fmt.Println("DONE 1 2 1 1 cancel")
					}
					continue
				}
				expected := map[string]string{
					"engine-topk-20":      "GEN 32 top_k=20 72,105",
					"engine-topk-64":      "GEN 32 top_k=64 72,105",
					"engine-topk-default": "GEN 32 72,105",
				}
				if want, ok := expected[os.Args[3]]; ok && scanner.Text() != want {
					fmt.Println("ERR unexpected top_k")
					continue
				}
				text := "hello STOPhidden"
				if os.Args[3] == "engine-tool" {
					text = "<tool_call><function=lookup><parameter=x>7</parameter></function></tool_call>"
				} else if os.Args[3] == "engine-deep-tool" {
					text = `<｜DSML｜ calls><｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="x" string="true">123</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`
				} else if os.Args[3] == "engine-deep-reason" {
					text = "fallback answer"
				}
				for _, b := range []byte(text) {
					fmt.Printf("T %d\n", b)
				}
				count := len(text)
				if strings.HasPrefix(os.Args[3], "engine-deep-") {
					fmt.Println("T 257")
					fmt.Println("T 104")
					count += 2
				}
				fmt.Printf("DONE %d 2 1 1 stop\n", count)
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
	vocab["<control>"] = 256
	vocab["<｜end▁of▁sentence｜>"] = 257
	vocab["｜DSML｜"] = 258
	types := make([]int, 259)
	types[256] = 3
	types[257] = 3
	types[258] = 4
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
	write("token_type.json", types)
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
	return setupEngine(t, "engine")
}

func setupEngine(t *testing.T, mode string) (http.Handler, *engine.Engine) {
	return setupFamilyEngine(t, mode, requestnorm.FamilyQwen)
}

func setupFamilyEngine(t *testing.T, mode string, family requestnorm.Family) (http.Handler, *engine.Engine) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tok, dir := syntheticTokenizer(t)
	renderer, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	eng, err := engine.Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", mode}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = eng.Close(ctx)
	})
	var servingEngine Engine = eng
	if mode == "engine-cancel" {
		supervisor, err := engine.Supervise(eng, func(ctx context.Context) (*engine.Engine, error) {
			return engine.Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "engine"}, io.Discard)
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = supervisor.Close(ctx)
		})
		servingEngine = supervisor
	}
	h, err := New(Config{Model: "test", Family: family, Tokenizer: tok, Engine: servingEngine, Renderer: renderer, APIKeys: []string{"secret"}, MaxBody: 4096, MaxPromptBytes: 2048, MaxOutputTokens: 200, MaxConcurrent: 2})
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
	status, _ = request(t, ts.Client(), url, `{"messages":[{"role":"user","content":"<control>"}]}`, "secret")
	if status != 502 {
		t.Fatalf("control literal status %d", status)
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

func TestServerRecoversAfterClientCancellation(t *testing.T) {
	h, _ := setupEngine(t, "engine-cancel")
	ts := httptest.NewServer(h)
	defer ts.Close()
	client := ts.Client()
	client.Timeout = 5 * time.Second
	url := ts.URL + "/v1/chat/completions"
	body := `{"messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":32,` +
		`"chat_template_kwargs":{"enable_thinking":false}}`
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer secret")
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(resp.Body)
	gotToken := false
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), `"content":"h"`) {
			gotToken = true
			break
		}
	}
	if !gotToken {
		resp.Body.Close()
		t.Fatalf("no first token: %v", scanner.Err())
	}
	cancel()
	resp.Body.Close()
	status, raw := request(t, client, url,
		`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}`,
		"secret")
	if status != http.StatusOK || !strings.Contains(raw, `"content":"hello STOPhidden"`) {
		t.Fatalf("request after disconnect: status %d, body %s", status, raw)
	}
}

func TestHTTPTopKReachesEngine(t *testing.T) {
	for _, tc := range []struct {
		name, value, engine string
		status              int
	}{
		{"integer", "20", "engine-topk-20", 200},
		{"numeric string", `"20.0"`, "engine-topk-20", 200},
		{"clamped", "128", "engine-topk-64", 200},
		{"large integer", "999999999999999999999999999999", "engine-topk-64", 200},
		{"zero", "0", "engine-topk-64", 200},
		{"negative", "-1", "engine-topk-default", 200},
		{"large negative integer", "-999999999999999999999999999999", "engine-topk-default", 200},
		{"null", "null", "engine-topk-default", 200},
		{"float", "20.0", "engine-topk-default", 400},
		{"boolean", "true", "engine-topk-default", 400},
		{"fractional string", `"20.1"`, "engine-topk-default", 400},
		{"hexadecimal string", `"0x1p2"`, "engine-topk-default", 400},
		{"nonfinite string", `"Infinity"`, "engine-topk-default", 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, _ := setupEngine(t, tc.engine)
			ts := httptest.NewServer(h)
			defer ts.Close()
			body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,` +
				`"chat_template_kwargs":{"enable_thinking":false},"top_k":` + tc.value + `}`
			status, raw := request(t, ts.Client(), ts.URL+"/v1/chat/completions", body, "secret")
			if status != tc.status {
				t.Fatalf("status %d, want %d: %s", status, tc.status, raw)
			}
			if status == 200 && !strings.Contains(raw, `"content":"hello STOPhidden"`) {
				t.Fatalf("response %s", raw)
			}
		})
	}
}

func TestRendererRequired(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	if _, err := NewTemplateRenderer(t.TempDir(), tok); err == nil {
		t.Fatal("missing renderer accepted")
	}
}

func TestStreamOptionsAllowUsageRequests(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	for _, option := range []string{`{"include_usage":true}`, `{"include_usage":false}`, `null`} {
		body := `{"messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":` + option +
			`,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}`
		status, raw := request(t, ts.Client(), ts.URL+"/v1/chat/completions", body, "secret")
		if status != http.StatusOK {
			t.Fatalf("stream_options %s: status %d: %s", option, status, raw)
		}
		var content strings.Builder
		var finish string
		var usage *struct {
			Prompt     int `json:"prompt_tokens"`
			Completion int `json:"completion_tokens"`
			Total      int `json:"total_tokens"`
		}
		done := false
		for _, line := range strings.Split(raw, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				done = true
				continue
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					Finish string `json:"finish_reason"`
				} `json:"choices"`
				Usage *struct {
					Prompt     int `json:"prompt_tokens"`
					Completion int `json:"completion_tokens"`
					Total      int `json:"total_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil || len(chunk.Choices) != 1 {
				t.Fatalf("invalid chunk %q: %v", data, err)
			}
			content.WriteString(chunk.Choices[0].Delta.Content)
			if chunk.Choices[0].Finish != "" {
				finish = chunk.Choices[0].Finish
			}
			if chunk.Usage != nil {
				usage = chunk.Usage
			}
		}
		if !done || content.String() != "hello STOPhidden" || finish != "stop" ||
			usage == nil || usage.Prompt != 2 || usage.Completion != 16 || usage.Total != 18 {
			t.Fatalf("stream_options %s: content %q, finish %q, usage %#v, done %t", option, content.String(), finish, usage, done)
		}
	}
}

func TestHTTPToolCall(t *testing.T) {
	h, _ := setupEngine(t, "engine-tool")
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":100,"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}}]}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/chat/completions", body, "secret")
	if status != http.StatusOK {
		t.Fatalf("status %d: %s", status, raw)
	}
	var response struct {
		Choices []struct {
			Finish  string `json:"finish_reason"`
			Message struct {
				Calls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(raw), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Finish != "tool_calls" || len(response.Choices[0].Message.Calls) != 1 {
		t.Fatalf("tool response: %s", raw)
	}
	call := response.Choices[0].Message.Calls[0]
	var arguments map[string]int
	if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil || call.ID == "" ||
		call.Function.Name != "lookup" || arguments["x"] != 7 {
		t.Fatalf("tool call: %#v, arguments: %v, error: %v", call, arguments, err)
	}
	streamBody := strings.TrimSuffix(body, "}") + `,"stream":true}`
	status, raw = request(t, ts.Client(), ts.URL+"/v1/chat/completions", streamBody, "secret")
	if status != http.StatusOK || !strings.Contains(raw, `"name":"lookup"`) ||
		!strings.Contains(raw, `"finish_reason":"tool_calls"`) || !strings.Contains(raw, "data: [DONE]") {
		t.Fatalf("streamed tool response %d: %s", status, raw)
	}
}

func TestDeepSeekToolCallAndStopToken(t *testing.T) {
	h, _ := setupFamilyEngine(t, "engine-deep-tool", requestnorm.FamilyDeepSeek)
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false},` +
		`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}}]}`
	url := ts.URL + "/v1/chat/completions"
	status, raw := request(t, ts.Client(), url, body, "secret")
	if status != http.StatusOK {
		t.Fatalf("status %d: %s", status, raw)
	}
	var response struct {
		Choices []struct {
			Finish  string `json:"finish_reason"`
			Message struct {
				Content *string `json:"content"`
				Calls   []struct {
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(raw), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Finish != "tool_calls" ||
		len(response.Choices[0].Message.Calls) != 1 {
		t.Fatalf("tool response: %s", raw)
	}
	call := response.Choices[0].Message.Calls[0]
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil ||
		call.Function.Name != "lookup" || args["x"] != "123" {
		t.Fatalf("tool call: %#v, arguments: %v, error: %v", call, args, err)
	}
	if response.Choices[0].Message.Content != nil && strings.Contains(*response.Choices[0].Message.Content, "h") {
		t.Fatalf("text after end token leaked: %s", raw)
	}
	streamBody := strings.TrimSuffix(body, "}") + `,"stream":true}`
	status, raw = request(t, ts.Client(), url, streamBody, "secret")
	if status != http.StatusOK || !strings.Contains(raw, `"name":"lookup"`) ||
		!strings.Contains(raw, `"finish_reason":"tool_calls"`) || !strings.Contains(raw, "data: [DONE]") {
		t.Fatalf("streamed tool response %d: %s", status, raw)
	}
}

func TestDeepSeekUnclosedThinkingBecomesAnswer(t *testing.T) {
	h, _ := setupFamilyEngine(t, "engine-deep-reason", requestnorm.FamilyDeepSeek)
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/chat/completions", body, "secret")
	if status != http.StatusOK {
		t.Fatalf("status %d: %s", status, raw)
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(raw), &response); err != nil || len(response.Choices) != 1 ||
		response.Choices[0].Message.Content != "fallback answer" || response.Choices[0].Message.Reasoning != "" {
		t.Fatalf("response %s: %v", raw, err)
	}
}

func TestModelListTracksEngine(t *testing.T) {
	h, eng := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	get := func(key string) *http.Response {
		t.Helper()
		req, err := http.NewRequest(http.MethodGet, ts.URL+"/v1/models", nil)
		if err != nil {
			t.Fatal(err)
		}
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := ts.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}
	resp := get("")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthorized status %d", resp.StatusCode)
	}
	resp.Body.Close()
	check := func(want int) {
		t.Helper()
		resp := get("secret")
		defer resp.Body.Close()
		var body struct {
			Object string `json:"object"`
			Data   []struct {
				ID   string `json:"id"`
				Meta struct {
					Context int `json:"n_ctx"`
				} `json:"meta"`
			} `json:"data"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK || body.Object != "list" || len(body.Data) != want {
			t.Fatalf("status %d, body %#v", resp.StatusCode, body)
		}
		if want == 1 && (body.Data[0].ID != "test" || body.Data[0].Meta.Context != 4096) {
			t.Fatalf("model %#v", body.Data[0])
		}
	}
	check(1)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := eng.Close(ctx); err != nil {
		t.Fatal(err)
	}
	check(0)
}

func TestHealthTracksEngineWithoutAPIKey(t *testing.T) {
	h, eng := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	check := func(path string, loaded bool) {
		t.Helper()
		resp, err := ts.Client().Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var body struct {
			Status          string `json:"status"`
			MaxContext      int    `json:"max_context"`
			Model           string `json:"model"`
			Images          bool   `json:"images"`
			APIKey          bool   `json:"api_key"`
			ImageGeneration bool   `json:"image_generation"`
			Loaded          bool   `json:"loaded"`
			Service         string `json:"service"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK || body.Status != "ok" || body.MaxContext != 4096 ||
			body.Model != "test" || body.Images || !body.APIKey || body.ImageGeneration ||
			body.Loaded != loaded || body.Service != "strata" {
			t.Fatalf("%s: status %d, body %#v", path, resp.StatusCode, body)
		}
	}
	check("/health", true)
	check("/api/health", true)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := eng.Close(ctx); err != nil {
		t.Fatal(err)
	}
	check("/health", false)
}
