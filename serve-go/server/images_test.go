package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"strata/engine"
	"strata/outputparser"
	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
)

type imageReader struct {
	dir   string
	paths []string
	fail  bool
}

func (v *imageReader) Encode(ctx context.Context, data []byte) (string, int, error) {
	deadline, ok := ctx.Deadline()
	if !ok || time.Until(deadline) > 300*time.Second {
		return "", 0, errors.New("missing image timeout")
	}
	if v.fail && len(v.paths) == 1 {
		return "", 0, errors.New("encode failed")
	}
	path := filepath.Join(v.dir, string(rune('a'+len(v.paths))))
	if err := os.WriteFile(path, data, 0600); err != nil {
		return "", 0, err
	}
	v.paths = append(v.paths, path)
	return path, 3, nil
}

type imageEngine struct {
	paths []string
	ids   []int
	data  string
}

func (*imageEngine) Alive() bool       { return true }
func (*imageEngine) ContextLimit() int { return 4096 }
func (e *imageEngine) GenerateWithOptions(_ context.Context, ids []int, _ int, o engine.RequestOptions) (engine.Result, error) {
	e.paths = append(e.paths, o.EmbeddingsPath)
	e.ids = append([]int(nil), ids...)
	if o.EmbeddingsPath != "" {
		data, err := os.ReadFile(o.EmbeddingsPath)
		if err != nil {
			return engine.Result{}, err
		}
		e.data = string(data)
	}
	if err := o.OnToken('a'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop"}, nil
}

func imageHandler(t *testing.T, vision *imageReader) (http.Handler, *imageEngine) {
	t.Helper()
	_, dir := syntheticTokenizer(t)
	var vocab map[string]int
	b, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &vocab); err != nil {
		t.Fatal(err)
	}
	vocab["<|vision_start|>"], vocab["<|image_pad|>"], vocab["<|vision_end|>"] = 260, 261, 262
	b, err = json.Marshal(vocab)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "vocab.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	types := make([]int, 263)
	types[256], types[257], types[258], types[260], types[261], types[262] = 3, 3, 4, 3, 3, 3
	b, err = json.Marshal(types)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "token_type.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	template := `{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}`
	if err = os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(template), 0600); err != nil {
		t.Fatal(err)
	}
	tok, err := modeltokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	renderer, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	e := &imageEngine{}
	cfg := Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: e, Renderer: renderer, APIKeys: []string{"secret"}, MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 4}
	if vision != nil {
		cfg.Vision = vision
	}
	h, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return h, e
}

func TestImageAPIs(t *testing.T) {
	image := `data:image/png;base64,QQ==`
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"text","text":"<|vision_start|><|image_pad|>"},{"type":"image_url","image_url":{"url":"` + image + `"}},{"type":"image_url","image_url":{"url":"` + image + `"}}]}],"chat_template_kwargs":{"enable_thinking":false}}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"` + image + `"}}]}],"max_tokens":32}`},
		{"/v1/responses", `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"` + image + `"}]}]}`},
	} {
		t.Run(tc.path, func(t *testing.T) {
			reader := &imageReader{dir: t.TempDir()}
			h, eng := imageHandler(t, reader)
			w := responsesRequest(h, tc.path, tc.body, "secret")
			if w.Code != 200 {
				t.Fatalf("%d %s", w.Code, w.Body.String())
			}
			if len(eng.paths) != 1 || eng.paths[0] == "" || eng.data != strings.Repeat("A", len(reader.paths)) {
				t.Fatalf("engine=%+v reader=%+v", eng, reader)
			}
			for _, path := range append(reader.paths, eng.paths...) {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("file remains: %s: %v", path, err)
				}
			}
			pads := 0
			for _, id := range eng.ids {
				if id == 261 {
					pads++
				}
			}
			if pads != 3*len(reader.paths) {
				t.Fatalf("image pad count %d, ids %v", pads, eng.ids)
			}
		})
	}
}

func TestForeignOriginLocalImages(t *testing.T) {
	local := filepath.Join(t.TempDir(), "image")
	if err := os.WriteFile(local, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"` + local + `"}}]}]}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"` + local + `"}}]}],"max_tokens":32}`},
		{"/v1/responses", `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"` + local + `"}]}]}`},
	} {
		t.Run(tc.path, func(t *testing.T) {
			reader := &imageReader{dir: t.TempDir()}
			h, _ := imageHandler(t, reader)
			r := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			r.Header.Set("Authorization", "Bearer secret")
			r.Header.Set("Origin", "https://foreign.example")
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != 400 || len(reader.paths) != 0 {
				t.Fatalf("%d %s paths=%v", w.Code, w.Body.String(), reader.paths)
			}
		})
	}
}

func TestImageFailureCleanupAndCount(t *testing.T) {
	reader := &imageReader{dir: t.TempDir(), fail: true}
	h, eng := imageHandler(t, reader)
	body := `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}},{"type":"image_url","image_url":{"url":"data:image/png;base64,Qg=="}}]}]}`
	w := responsesRequest(h, "/v1/chat/completions", body, "secret")
	if w.Code != 400 || len(eng.paths) != 0 || len(reader.paths) != 1 {
		t.Fatalf("status=%d body=%s reader=%+v engine=%+v", w.Code, w.Body.String(), reader, eng)
	}
	if _, err := os.Stat(reader.paths[0]); !os.IsNotExist(err) {
		t.Fatalf("failed request left image: %v", err)
	}
	count := `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"not-a-file"}}]}]}`
	w = responsesRequest(h, "/v1/messages/count_tokens", count, "secret")
	if w.Code != 200 || len(reader.paths) != 1 {
		t.Fatalf("count status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestImageStreamAndDisabled(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}}]}],"stream":true}`
	reader := &imageReader{dir: t.TempDir()}
	h, eng := imageHandler(t, reader)
	w := responsesRequest(h, "/v1/chat/completions", body, "secret")
	if w.Code != 200 || !strings.Contains(w.Body.String(), "[DONE]") || len(eng.paths) != 1 {
		t.Fatalf("stream status=%d body=%s", w.Code, w.Body.String())
	}
	h, eng = imageHandler(t, nil)
	w = responsesRequest(h, "/v1/chat/completions", body, "secret")
	if w.Code != 400 || len(eng.paths) != 0 {
		t.Fatalf("disabled status=%d body=%s", w.Code, w.Body.String())
	}
}

type continuationImageEngine struct {
	budgetEngine
	paths []string
}

func (e *continuationImageEngine) GenerateWithOptions(ctx context.Context, ids []int, max int, opts engine.RequestOptions) (engine.Result, error) {
	e.paths = append(e.paths, opts.EmbeddingsPath)
	return e.budgetEngine.GenerateWithOptions(ctx, ids, max, opts)
}

func TestImageEmbeddingsOnContinuation(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	parser, err := outputparser.New(outputparser.Options{Thinking: true})
	if err != nil {
		t.Fatal(err)
	}
	e := &continuationImageEngine{budgetEngine: budgetEngine{segments: [][]int{bytesAsTokens("thought"), bytesAsTokens("answer")}}}
	_, err = generateWithEmbeddings(context.Background(), e, tok, parser, []int{'H'}, 300, engine.Sampling{}, nil, 3, "", func([]outputparser.Event) error { return nil }, "image-file", nil, nil)
	if err != nil || len(e.paths) != 2 || e.paths[0] != "image-file" || e.paths[1] != "image-file" {
		t.Fatalf("paths=%v err=%v", e.paths, err)
	}
}

type gateEngine struct {
	entered chan struct{}
	release chan struct{}
}

func (*gateEngine) Alive() bool       { return true }
func (*gateEngine) ContextLimit() int { return 4096 }
func (e *gateEngine) GenerateWithOptions(ctx context.Context, _ []int, _ int, _ engine.RequestOptions) (engine.Result, error) {
	e.entered <- struct{}{}
	select {
	case <-e.release:
		return engine.Result{Finish: "stop"}, nil
	case <-ctx.Done():
		return engine.Result{}, ctx.Err()
	}
}

func TestVisionGateHonorsCancellation(t *testing.T) {
	tok, dir := syntheticTokenizer(t)
	renderer, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	e := &gateEngine{entered: make(chan struct{}, 2), release: make(chan struct{})}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: e,
		Renderer: renderer, Vision: &imageReader{dir: t.TempDir()}, MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 4})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"messages":[{"role":"user","content":"hello"}]}`
	done := make(chan struct{})
	go func() {
		responsesRequest(h, "http://127.0.0.1/v1/chat/completions", body, "")
		close(done)
	}()
	select {
	case <-e.entered:
	case <-done:
		t.Fatal("first request returned before entering engine")
	case <-time.After(5 * time.Second):
		t.Fatal("first request did not enter engine")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := httptest.NewRequest(http.MethodPost, "http://127.0.0.1/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
	h.ServeHTTP(httptest.NewRecorder(), r)
	select {
	case <-e.entered:
		t.Fatal("cancelled request entered engine")
	default:
	}
	close(e.release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("first request did not complete")
	}
}
