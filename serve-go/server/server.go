// Package server wires the existing Go components into an opt-in, text-only
// OpenAI Chat endpoint. It is not a replacement for the Python server.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

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
	Engine          Engine
	Renderer        Renderer
	APIKeys         []string
	MaxBody         int64
	MaxPromptBytes  int
	MaxOutputTokens int
	MaxConcurrent   int
}

// Engine is the native generation service used by the handler.
type Engine interface {
	Alive() bool
	ContextLimit() int
	GenerateWithOptions(context.Context, []int, int, engine.RequestOptions) (engine.Result, error)
}

type server struct {
	cfg        Config
	normalizer *requestnorm.Normalizer
	stopIDs    map[int]bool
}

func New(c Config) (http.Handler, error) {
	if c.Model == "" || (c.Family != requestnorm.FamilyQwen && c.Family != requestnorm.FamilyDeepSeek) || c.Tokenizer == nil || c.Engine == nil || c.Renderer == nil ||
		c.MaxBody <= 0 || c.MaxPromptBytes <= 0 || c.MaxOutputTokens <= 0 || c.MaxConcurrent <= 0 {
		return nil, errors.New("invalid server configuration")
	}
	n, err := requestnorm.NewNormalizer(c.Family)
	if err != nil {
		return nil, err
	}
	s := &server{cfg: c, normalizer: n, stopIDs: map[int]bool{}}
	stopTags := []string{"<|im_end|>", "<|endoftext|>"}
	if c.Family == requestnorm.FamilyDeepSeek {
		stopTags = []string{"<｜end▁of▁sentence｜>"}
	}
	for _, tag := range stopTags {
		special, e := c.Tokenizer.Encode(tag, true, nil)
		plain, e2 := c.Tokenizer.Encode(tag, false, nil)
		if e == nil && e2 == nil && len(special) == 1 && (len(plain) != 1 || special[0] != plain[0]) {
			s.stopIDs[special[0]] = true
		}
	}
	guard, err := httpguard.New(httpguard.Config{APIKeys: c.APIKeys, MaxBody: c.MaxBody, AllowHealth: true, AllowModels: true}, http.HandlerFunc(s.route))
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
	if r.URL.Path == "/health" || r.URL.Path == "/api/health" {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "ok", "max_context": s.cfg.Engine.ContextLimit(), "model": s.cfg.Model,
			"images": false, "api_key": len(s.cfg.APIKeys) > 0, "image_generation": false,
			"loaded": s.cfg.Engine.Alive(), "service": "strata",
		})
		return
	}
	if r.URL.Path == "/v1/models" || r.URL.Path == "/models" {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			fail(w, http.StatusMethodNotAllowed, "invalid_request_error", "GET required")
			return
		}
		models := []any{}
		if s.cfg.Engine.Alive() {
			models = append(models, map[string]any{
				"id": s.cfg.Model, "object": "model", "status": map[string]string{"value": "loaded"},
				"meta":         map[string]int{"n_ctx": s.cfg.Engine.ContextLimit()},
				"architecture": map[string]any{"input_modalities": []string{"text"}, "output_modalities": []string{"text"}},
			})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": models})
		return
	}
	countTokens := r.URL.Path == "/v1/messages/count_tokens" || r.URL.Path == "/v1/v1/messages/count_tokens"
	messages := r.URL.Path == "/v1/messages" || r.URL.Path == "/v1/v1/messages"
	responses := r.URL.Path == "/v1/responses"
	if r.URL.Path != "/v1/chat/completions" && !countTokens && !messages && !responses {
		fail(w, http.StatusNotImplemented, "unsupported_endpoint", "endpoint is not implemented")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		fail(w, http.StatusMethodNotAllowed, "invalid_request_error", "POST required")
		return
	}
	if responses {
		s.responses(w, r)
	} else if countTokens {
		s.countTokens(w, r)
	} else if messages {
		s.anthropic(w, r)
	} else {
		s.chat(w, r)
	}
}

type options struct {
	stream     bool
	noTools    bool
	force      string
	max        int
	stops      []string
	sampling   engine.Sampling
	structured bool
	repair     bool
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
	o.stops, err = parseStops(root)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	var toolChoice requestnorm.Value
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
		case "tool_choice":
			toolChoice = v
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
		case "stream_options":
			// Python ignores this field and always includes usage in the final chunk.
		case "user", "metadata", "service_tier", "safety_identifier", "prompt_cache_key":
			// Python accepts these client metadata fields without changing generation.
		case "response_format":
			if v.Kind() != requestnorm.Null {
				kind, ok := v.Lookup("type")
				text, isString := kind.Text()
				if v.Kind() != requestnorm.Object || !ok || !isString || kind.Kind() != requestnorm.String || (text != "text" && text != "json_object") {
					return requestnorm.Result{}, o, errors.New("structured response_format is not supported")
				}
				if text == "json_object" {
					o.structured = true
					repair, _ := v.Lookup("repair_latex_escapes")
					o.repair, _ = repair.Bool()
				}
			}
		case "max_tokens", "max_completion_tokens":
			if maxSet {
				return requestnorm.Result{}, o, errors.New("provide only one max token field")
			}
			var n int
			b, _ := v.MarshalJSON()
			if json.Unmarshal(b, &n) != nil || n > s.cfg.MaxOutputTokens {
				return requestnorm.Result{}, o, errors.New("invalid max tokens")
			}
			if n > 0 {
				o.max = n
			}
			maxSet = true
		case "stop", "stop_sequences":
		case "temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty",
			"penalty_last_n", "seed", "repeat_penalty", "repeat_last_n":
			if err := setSampling(&o.sampling, root, field.Name, v); err != nil {
				return requestnorm.Result{}, o, err
			}
		case "top_k":
			if v.Kind() == requestnorm.Null {
				continue
			}
			var k int
			if text, ok := v.Integer(); ok {
				n, err := strconv.ParseInt(text, 10, 64)
				if err != nil {
					if strings.HasPrefix(text, "-") {
						continue
					}
					k = 64
				} else if n < 0 {
					continue
				} else if n == 0 || n > 64 {
					k = 64
				} else {
					k = int(n)
				}
			} else if text, ok := v.Text(); ok && v.Kind() == requestnorm.String {
				text = strings.TrimSpace(text)
				n, err := strconv.ParseFloat(text, 64)
				if err != nil || math.IsInf(n, 0) || math.IsNaN(n) || math.Trunc(n) != n ||
					strings.HasPrefix(strings.ToLower(strings.TrimLeft(text, "+-")), "0x") {
					return requestnorm.Result{}, o, errors.New("top_k must be an integer")
				}
				if n < 0 {
					continue
				}
				if n == 0 || n > 64 {
					k = 64
				} else {
					k = int(n)
				}
			} else {
				return requestnorm.Result{}, o, errors.New("top_k must be an integer")
			}
			o.sampling.TopK = &k
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
	if o.structured {
		result.Messages = addObjectDirective(result.Messages, false)
	}
	kind, force := normalizeForceChoice(toolChoice, result.Tools, s.cfg.Family)
	o.noTools, o.force = kind == "none", force
	if o.noTools {
		result.Tools = nil
	}
	if o.structured && len(result.Tools) > 0 {
		return requestnorm.Result{}, o, errors.New("structured response_format with tools is not supported")
	}
	return result, o, nil
}

var stopGeneration = errors.New("stop sequence reached")

// The native engine reserves eight context positions. Match Python's default
// fit_max_tokens behavior by shortening the answer to the space left.
func fitOutputTokens(contextLimit, promptTokens, requested int) (int, error) {
	room := contextLimit - 8 - promptTokens
	if room < 1 {
		return 0, errors.New("prompt leaves no room for output")
	}
	if requested > room {
		return room, nil
	}
	return requested, nil
}

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
	ids, err = appendForcedPrompt(s.cfg.Tokenizer, rendered, ids, opt.force, s.cfg.MaxPromptBytes)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	opt.max, err = fitOutputTokens(s.cfg.Engine.ContextLimit(), len(ids), opt.max)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
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
	parserOptions := outputparser.Options{Thinking: rendered.Thinking, Tools: tools, DeclaredOnly: true}
	var parser generationParser
	if s.cfg.Family == requestnorm.FamilyDeepSeek {
		parser, err = outputparser.NewDeepSeek(parserOptions)
	} else {
		parser, err = outputparser.New(parserOptions)
	}
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	var opening []outputparser.Event
	if opt.force != "" {
		opening, err = parser.Feed(opt.force)
		if err != nil {
			fail(w, 400, "invalid_request_error", err.Error())
			return
		}
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
	started := false
	structuredReady := false
	var timings map[string]any
	send := func(delta map[string]any, finish any, usage any) error {
		if !opt.stream || (opt.structured && !structuredReady) {
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
			msg["timings"] = timings
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
	err = route(opening)
	var pumped generationResult
	if err == nil {
		pumped, err = generate(r.Context(), s.cfg.Engine, s.cfg.Tokenizer, parser, ids, opt.max, opt.sampling, s.stopIDs, route)
	}
	result, count := pumped.result, pumped.count
	timings = pumped.timings
	finish := result.Finish
	if pumped.stopped {
		finish = "stop"
	}
	if err == nil {
		err = emitContent(matcher.Finish())
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
	if opt.structured {
		validated, validationErr := validateJSONObject(content.String(), finish, opt.repair)
		if validationErr != nil {
			body, _ := json.Marshal(map[string]any{"error": map[string]string{
				"type": "structured_output_failed", "code": "structured_output_failed", "message": validationErr.Error(),
			}})
			if opt.stream {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", body)
				_ = http.NewResponseController(w).Flush()
			} else {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadGateway)
				_, _ = w.Write(body)
			}
			return
		}
		content.Reset()
		content.WriteString(validated)
		structuredReady = true
		if opt.stream {
			if err = send(map[string]any{"role": "assistant", "content": ""}, nil, nil); err != nil {
				return
			}
			delta := map[string]any{"content": validated}
			if reason.Len() > 0 {
				delta["reasoning_content"] = reason.String()
			}
			if err = send(delta, nil, nil); err != nil {
				return
			}
		}
	}
	if deep, ok := parser.(*outputparser.DeepSeekParser); ok && deep.ReasoningFallback() && !opt.stream {
		reason.Reset()
	}
	usage := map[string]any{"prompt_tokens": len(ids), "completion_tokens": count, "total_tokens": len(ids) + count,
		"prompt_tokens_details": map[string]int{"cached_tokens": pumped.cached}}
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
	_ = json.NewEncoder(w).Encode(map[string]any{"id": id, "object": "chat.completion", "created": created, "model": s.cfg.Model, "choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}}, "usage": usage, "timings": timings})
}
