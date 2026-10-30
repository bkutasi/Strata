// Package server wires the existing Go components into an opt-in, text-only
// OpenAI Chat endpoint. It is not a replacement for the Python server.
package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"strata.local/httpguard"
	"strata/engine"
	"strata/outputparser"
	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
	"strata/streamtext"
)

type Config struct {
	Model           string
	Family          requestnorm.Family
	Tokenizer       *modeltokenizer.Tokenizer
	Engine          *engine.Engine
	Renderer        Renderer
	APIKeys         []string
	MaxBody         int64
	MaxPromptBytes  int
	MaxOutputTokens int
	MaxConcurrent   int
}

type server struct {
	cfg        Config
	normalizer *requestnorm.Normalizer
	stopIDs    map[int]bool
}

func New(c Config) (http.Handler, error) {
	if c.Model == "" || c.Family != requestnorm.FamilyQwen || c.Tokenizer == nil || c.Engine == nil || c.Renderer == nil ||
		c.MaxBody <= 0 || c.MaxPromptBytes <= 0 || c.MaxOutputTokens <= 0 || c.MaxConcurrent <= 0 {
		return nil, errors.New("invalid server configuration; only qwen text is supported")
	}
	n, err := requestnorm.NewNormalizer(c.Family)
	if err != nil {
		return nil, err
	}
	s := &server{cfg: c, normalizer: n, stopIDs: map[int]bool{}}
	for _, tag := range []string{"<|im_end|>", "<|endoftext|>"} {
		special, e := c.Tokenizer.Encode(tag, true, nil)
		plain, e2 := c.Tokenizer.Encode(tag, false, nil)
		if e == nil && e2 == nil && len(special) == 1 && (len(plain) != 1 || special[0] != plain[0]) {
			s.stopIDs[special[0]] = true
		}
	}
	guard, err := httpguard.New(httpguard.Config{APIKeys: c.APIKeys, MaxBody: c.MaxBody}, http.HandlerFunc(s.route))
	if err != nil {
		return nil, err
	}
	// Admission precedes buffering in httpguard.
	sem := make(chan struct{}, c.MaxConcurrent)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
			guard.ServeHTTP(w, r)
		default:
			fail(w, http.StatusTooManyRequests, "rate_limit_error", "server admission is full")
		}
	}), nil
}

func fail(w http.ResponseWriter, status int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"type": kind, "message": msg}})
}

func (s *server) route(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/models" {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			fail(w, http.StatusMethodNotAllowed, "invalid_request_error", "GET required")
			return
		}
		models := []any{}
		if s.cfg.Engine.Alive() {
			models = append(models, map[string]any{
				"id": s.cfg.Model, "object": "model", "status": map[string]string{"value": "loaded"},
				"meta":         map[string]int{"n_ctx": s.cfg.Engine.Context},
				"architecture": map[string]any{"input_modalities": []string{"text"}, "output_modalities": []string{"text"}},
			})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": models})
		return
	}
	if r.URL.Path != "/v1/chat/completions" {
		fail(w, http.StatusNotImplemented, "unsupported_endpoint", "endpoint is not implemented")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		fail(w, http.StatusMethodNotAllowed, "invalid_request_error", "POST required")
		return
	}
	s.chat(w, r)
}

type options struct {
	stream   bool
	max      int
	stops    []string
	sampling engine.Sampling
}

func (s *server) decode(data []byte) (requestnorm.Result, options, error) {
	root, err := requestnorm.ParseValue(data)
	if err != nil {
		return requestnorm.Result{}, options{}, err
	}
	if root.Kind() != requestnorm.Object {
		return requestnorm.Result{}, options{}, errors.New("request must be an object")
	}
	promptFields := map[string]json.RawMessage{}
	o := options{max: s.cfg.MaxOutputTokens}
	maxSet := false
	for _, field := range root.Fields() {
		v := field.Value
		switch field.Name {
		case "messages", "tools", "reasoning_effort", "reasoning", "chat_template_kwargs":
			b, e := v.MarshalJSON()
			if e != nil {
				return requestnorm.Result{}, o, e
			}
			promptFields[field.Name] = b
		case "model":
			m, ok := v.Text()
			if !ok || v.Kind() != requestnorm.String || m != s.cfg.Model {
				return requestnorm.Result{}, o, errors.New("model not found")
			}
		case "stream":
			b, ok := v.Bool()
			if !ok {
				return requestnorm.Result{}, o, errors.New("stream must be a boolean")
			}
			o.stream = b
		case "max_tokens", "max_completion_tokens":
			if maxSet {
				return requestnorm.Result{}, o, errors.New("provide only one max token field")
			}
			var n int
			b, _ := v.MarshalJSON()
			if json.Unmarshal(b, &n) != nil || n < 1 || n > s.cfg.MaxOutputTokens {
				return requestnorm.Result{}, o, errors.New("invalid max tokens")
			}
			o.max, maxSet = n, true
		case "stop":
			if v.Kind() == requestnorm.String {
				t, _ := v.Text()
				o.stops = []string{t}
			} else if v.Kind() == requestnorm.Array {
				for _, item := range v.Items() {
					t, ok := item.Text()
					if !ok || item.Kind() != requestnorm.String {
						return requestnorm.Result{}, o, errors.New("stop must contain strings")
					}
					o.stops = append(o.stops, t)
				}
			} else {
				return requestnorm.Result{}, o, errors.New("stop must be a string or array")
			}
			if len(o.stops) > 4 {
				return requestnorm.Result{}, o, errors.New("at most four stop strings")
			}
			for _, t := range o.stops {
				if !utf8.ValidString(t) || len(t) > 1024 {
					return requestnorm.Result{}, o, errors.New("invalid stop string")
				}
			}
		case "temperature", "top_p":
			b, _ := v.MarshalJSON()
			var f float64
			if v.Kind() != requestnorm.Number || json.Unmarshal(b, &f) != nil || math.IsNaN(f) || math.IsInf(f, 0) {
				return requestnorm.Result{}, o, errors.New("invalid sampling value")
			}
			if field.Name == "temperature" {
				if f < 0 || f > 2 {
					return requestnorm.Result{}, o, errors.New("temperature must be between 0 and 2")
				}
				o.sampling.Temperature = &f
			} else {
				if f <= 0 || f > 1 {
					return requestnorm.Result{}, o, errors.New("top_p must be greater than 0 and at most 1")
				}
				o.sampling.TopP = &f
			}
		default:
			return requestnorm.Result{}, o, fmt.Errorf("unsupported request field %q", field.Name)
		}
	}
	b, err := json.Marshal(promptFields)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	req, err := requestnorm.DecodeOpenAI(b)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	for _, m := range req.Messages {
		for _, p := range m.Content.Parts {
			if p.Type == "image" {
				return requestnorm.Result{}, o, errors.New("images are not supported")
			}
		}
	}
	result, err := s.normalizer.NormalizeOpenAI(req)
	if err != nil {
		return result, o, err
	}
	if err = requestnorm.RequireMessages(result.Messages); err != nil {
		return result, o, err
	}
	return result, o, nil
}

var stopGeneration = errors.New("stop sequence reached")

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		fail(w, 400, "invalid_request_error", "invalid body")
		return
	}
	input, opt, err := s.decode(data)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	rendered, err := s.cfg.Renderer.Render(r.Context(), input)
	if err != nil {
		fail(w, 502, "server_error", "renderer failed")
		return
	}
	if len(rendered.Prompt) > s.cfg.MaxPromptBytes {
		fail(w, 400, "invalid_request_error", "prompt too large")
		return
	}
	ids, err := s.cfg.Tokenizer.EncodeWithBudget(rendered.Prompt, true, rendered.Plain, s.cfg.MaxPromptBytes)
	if err != nil || len(ids) == 0 {
		fail(w, 400, "invalid_request_error", "invalid rendered prompt")
		return
	}
	if len(ids)+opt.max > s.cfg.Engine.Context {
		fail(w, 400, "invalid_request_error", "prompt and output exceed context")
		return
	}
	var tools []map[string]any
	for _, v := range input.Tools {
		b, e := v.MarshalJSON()
		if e != nil {
			fail(w, 400, "invalid_request_error", "invalid tool")
			return
		}
		var m map[string]any
		dec := json.NewDecoder(strings.NewReader(string(b)))
		dec.UseNumber()
		if e = dec.Decode(&m); e != nil {
			fail(w, 400, "invalid_request_error", "invalid tool")
			return
		}
		tools = append(tools, m)
	}
	parser, err := outputparser.New(outputparser.Options{Thinking: rendered.Thinking, Tools: tools, DeclaredOnly: true})
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	var random [12]byte
	if _, err = rand.Read(random[:]); err != nil {
		fail(w, 500, "server_error", "cannot generate response ID")
		return
	}
	id, created := "chatcmpl-"+hex.EncodeToString(random[:]), time.Now().Unix()
	var content, reason strings.Builder
	var calls []map[string]any
	matcher := streamtext.NewStopMatcher(opt.stops)
	var decoder streamtext.Decoder
	started := false
	send := func(delta map[string]any, finish any, usage any) error {
		if !opt.stream {
			return nil
		}
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			started = true
		}
		msg := map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.cfg.Model, "choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
		if usage != nil {
			msg["usage"] = usage
		}
		b, e := json.Marshal(msg)
		if e != nil {
			return e
		}
		if _, e = fmt.Fprintf(w, "data: %s\n\n", b); e != nil {
			return e
		}
		return http.NewResponseController(w).Flush()
	}
	emitContent := func(text string) error {
		if text == "" {
			return nil
		}
		content.WriteString(text)
		return send(map[string]any{"content": text}, nil, nil)
	}
	route := func(events []outputparser.Event) error {
		for _, ev := range events {
			if ev.Kind == "content" {
				if e := emitContent(matcher.Push(ev.Text)); e != nil {
					return e
				}
				if matcher.Hit() != "" {
					return stopGeneration
				}
				continue
			}
			if e := emitContent(matcher.Boundary()); e != nil {
				return e
			}
			switch ev.Kind {
			case "reasoning":
				reason.WriteString(ev.Text)
				if ev.Text != "" {
					if e := send(map[string]any{"reasoning_content": ev.Text}, nil, nil); e != nil {
						return e
					}
				}
			case "tool_call":
				args, e := json.Marshal(ev.Call.Arguments)
				if e != nil {
					return e
				}
				call := map[string]any{"id": ev.Call.ID, "type": "function", "function": map[string]any{"name": ev.Call.Name, "arguments": string(args)}}
				index := len(calls)
				calls = append(calls, call)
				delta := map[string]any{"index": index, "id": ev.Call.ID, "type": "function", "function": call["function"]}
				if e = send(map[string]any{"tool_calls": []any{delta}}, nil, nil); e != nil {
					return e
				}
			}
		}
		return nil
	}
	if opt.stream {
		if err = send(map[string]any{"role": "assistant", "content": ""}, nil, nil); err != nil {
			return
		}
	}
	count := 0
	stoppedToken := false
	result, err := s.cfg.Engine.GenerateWithOptions(r.Context(), ids, opt.max, engine.RequestOptions{Sampling: opt.sampling, OnToken: func(id int) error {
		count++
		if count > opt.max {
			return errors.New("engine exceeded output token limit")
		}
		if stoppedToken {
			return nil
		}
		if s.stopIDs[id] {
			stoppedToken = true
			return nil
		}
		raw, e := s.cfg.Tokenizer.TokenBytes(id)
		if e != nil {
			return e
		}
		events, e := parser.Feed(decoder.Push(raw))
		if e != nil {
			return e
		}
		e = route(events)
		if errors.Is(e, stopGeneration) {
			// shortcut: drain to DONE instead of stopping inference, until reusable STOP is supported.
			stoppedToken = true
			return nil
		}
		return e
	}})
	finish := result.Finish
	if stoppedToken {
		finish = "stop"
	}
	if errors.Is(err, stopGeneration) {
		finish = "stop"
		err = nil
	}
	if err == nil {
		var events []outputparser.Event
		events, err = parser.Finish(finish)
		if err == nil {
			err = route(events)
			if errors.Is(err, stopGeneration) {
				err = nil
				finish = "stop"
			}
		}
		if err == nil {
			err = emitContent(matcher.Finish())
		}
	}
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		if !started {
			fail(w, 503, "server_error", "generation failed; restart server if engine exited")
			return
		}
		b, _ := json.Marshal(map[string]any{"error": map[string]string{"type": "server_error", "message": "generation failed; restart server if engine exited"}})
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", b)
		_ = http.NewResponseController(w).Flush()
		return
	}
	if finish == "cancel" {
		finish = "stop"
	}
	if len(calls) > 0 && finish == "stop" {
		finish = "tool_calls"
	}
	usage := map[string]int{"prompt_tokens": len(ids), "completion_tokens": count, "total_tokens": len(ids) + count}
	if opt.stream {
		if send(map[string]any{}, finish, usage) == nil {
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			_ = http.NewResponseController(w).Flush()
		}
		return
	}
	msg := map[string]any{"role": "assistant", "content": any(nil)}
	if content.Len() > 0 {
		msg["content"] = content.String()
	}
	if reason.Len() > 0 {
		msg["reasoning_content"] = reason.String()
	}
	if len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"id": id, "object": "chat.completion", "created": created, "model": s.cfg.Model, "choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}}, "usage": usage})
}
