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
	Model     string
	Aliases   []string
	Family    requestnorm.Family
	Tokenizer *modeltokenizer.Tokenizer
	Engine    Engine
	Renderer  Renderer
	Vision    interface {
		Encode(context.Context, []byte) (string, int, error)
	}
	APIKeys                    []string
	MaxBody                    int64
	MaxPromptBytes             int
	MaxOutputTokens            int
	MaxConcurrent              int
	DefaultReasoningBudget     *int
	DisableAnswerRoom          bool
	DisableFitMaxTokens        bool
	DisableResponses           bool
	DisableAnthropic           bool
	RepeatStopTokens           *int
	AnthropicThinkingOnRequest bool
	DefaultSampling            engine.Sampling
	DefaultReasoningEffort     string
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
	repeatStop int
	visionGate chan struct{}
}

func New(c Config) (http.Handler, error) {
	if c.Model == "" || (c.Family != requestnorm.FamilyQwen && c.Family != requestnorm.FamilyDeepSeek) || c.Tokenizer == nil || c.Engine == nil || c.Renderer == nil ||
		c.MaxBody <= 0 || c.MaxPromptBytes <= 0 || c.MaxOutputTokens <= 0 || c.MaxConcurrent <= 0 ||
		(c.RepeatStopTokens != nil && *c.RepeatStopTokens < 0) {
		return nil, errors.New("invalid server configuration")
	}
	c.Aliases = normalizeAliases(c.Model, c.Aliases)
	n, err := requestnorm.NewNormalizer(c.Family)
	if err != nil {
		return nil, err
	}
	s := &server{cfg: c, normalizer: n, stopIDs: map[int]bool{}, repeatStop: 256}
	if c.Vision != nil {
		s.visionGate = make(chan struct{}, 1)
	}
	if c.RepeatStopTokens != nil {
		s.repeatStop = *c.RepeatStopTokens
	}
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
			"images": s.cfg.Vision != nil, "api_key": len(s.cfg.APIKeys) > 0, "image_generation": false,
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
			modalities := []string{"text"}
			if s.cfg.Vision != nil {
				modalities = append(modalities, "image")
			}
			model := map[string]any{
				"id": s.cfg.Model, "object": "model", "status": map[string]string{"value": "loaded"},
				"meta":         map[string]int{"n_ctx": s.cfg.Engine.ContextLimit()},
				"architecture": map[string]any{"input_modalities": modalities, "output_modalities": []string{"text"}},
			}
			if len(s.cfg.Aliases) > 0 {
				model["aliases"] = s.cfg.Aliases
			}
			models = append(models, model)
			for _, alias := range s.cfg.Aliases {
				aliasModel := make(map[string]any, len(model)+1)
				for key, value := range model {
					aliasModel[key] = value
				}
				aliasModel["id"] = alias
				aliasModel["alias_of"] = s.cfg.Model
				models = append(models, aliasModel)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": models})
		return
	}
	path := strings.TrimSuffix(r.URL.Path, "/")
	countTokens := path == "/v1/messages/count_tokens" || path == "/v1/v1/messages/count_tokens"
	messages := path == "/v1/messages" || path == "/v1/v1/messages"
	responses := path == "/v1/responses"
	if (responses && s.cfg.DisableResponses) || ((messages || countTokens) && s.cfg.DisableAnthropic) {
		http.NotFound(w, r)
		return
	}
	if path != "/v1/chat/completions" && !countTokens && !messages && !responses {
		fail(w, http.StatusNotImplemented, "unsupported_endpoint", "endpoint is not implemented")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		fail(w, http.StatusMethodNotAllowed, "invalid_request_error", "POST required")
		return
	}
	if s.visionGate != nil {
		select {
		case s.visionGate <- struct{}{}:
			defer func() { <-s.visionGate }()
		case <-r.Context().Done():
			return
		}
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
	stream      bool
	noTools     bool
	force       string
	max         int
	explicitMax bool
	stops       []string
	sampling    engine.Sampling
	structured  bool
	repair      bool
	schema      *outputSchema
	budget      int
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
	budget, _ := root.Lookup("reasoning_budget_tokens")
	o.budget, err = reasoningBudget(root, budget)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	o.stops, err = parseStops(root)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	if _, present := root.Lookup("logprobs"); present {
		o.sampling.Logprobs, o.sampling.AllowedTokenIDs, err = chatLogprobs(root, s.cfg.Tokenizer.VocabularySize())
		if err != nil {
			return requestnorm.Result{}, o, err
		}
	}
	var toolChoice requestnorm.Value
	maxValue, _ := root.Lookup("max_completion_tokens")
	fallback := maxValue.Kind() == requestnorm.Null
	switch maxValue.Kind() {
	case requestnorm.Boolean:
		b, _ := maxValue.Bool()
		fallback = !b
	case requestnorm.Number:
		if integer, ok := maxValue.Integer(); ok {
			fallback = integer == "0"
		} else if n, ok := maxValue.Float64(); ok {
			fallback = n == 0
		}
	case requestnorm.String:
		text, _ := maxValue.Text()
		fallback = text == ""
	case requestnorm.Array:
		fallback = len(maxValue.Items()) == 0
	case requestnorm.Object:
		fallback = len(maxValue.Fields()) == 0
	}
	if fallback {
		maxValue, _ = root.Lookup("max_tokens")
	}
	if maxValue.Kind() != requestnorm.Null {
		var n int
		b, e := maxValue.MarshalJSON()
		if e != nil || json.Unmarshal(b, &n) != nil || n > s.cfg.MaxOutputTokens {
			return requestnorm.Result{}, o, errors.New("invalid max tokens")
		}
		if n > 0 {
			o.max = n
			o.explicitMax = true
		}
	}
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
			// Python serves the loaded model even when the requested name differs.
		case "stream":
			b, ok := v.Bool()
			if !ok {
				return requestnorm.Result{}, o, errors.New("stream must be a boolean")
			}
			o.stream = b
		case "stream_options":
			// Python ignores this field and always includes usage in the final chunk.
		case "user", "metadata", "service_tier", "safety_identifier", "prompt_cache_key", "n", "parallel_tool_calls":
			// Python accepts these fields without changing generation.
		case "response_format":
			if v.Kind() != requestnorm.Null {
				kind, ok := v.Lookup("type")
				text, isString := kind.Text()
				if v.Kind() != requestnorm.Object || !ok || !isString || kind.Kind() != requestnorm.String || (text != "text" && text != "json_object" && text != "json_schema") {
					return requestnorm.Result{}, o, errors.New("structured response_format is not supported")
				}
				if text == "json_object" || text == "json_schema" {
					o.structured = true
					repair, _ := v.Lookup("repair_latex_escapes")
					o.repair, _ = repair.Bool()
				}
				if text == "json_schema" {
					o.schema, err = compileOutputSchema(v)
					if err != nil {
						return requestnorm.Result{}, o, err
					}
				}
			}
		case "max_tokens", "max_completion_tokens":
		case "reasoning_budget_tokens", "thinking_budget_tokens":
		case "stop", "stop_sequences":
		case "strata_checkpoint":
			value, ok := v.Bool()
			o.sampling.DisableCheckpoint = ok && !value
		case "strata_tune":
			setTuning(&o.sampling, parseTuning(v, false))
		case "experimental_speed_projection":
			if value, ok := v.Bool(); ok {
				o.sampling.SpeedProjection = &value
			}
		case "logprobs", "top_logprobs", "allowed_token_ids":
		case "logit_bias":
			o.sampling.LogitBias, err = s.requestLogitBias(v)
			if err != nil {
				return requestnorm.Result{}, o, err
			}
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
	applySamplingDefaults(&o.sampling, s.cfg.DefaultSampling, root)
	b, err := json.Marshal(promptFields)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	req, err := requestnorm.DecodeOpenAI(b)
	if err != nil {
		return requestnorm.Result{}, o, err
	}
	if s.cfg.DefaultReasoningEffort != "" {
		effort, _ := root.Lookup("reasoning_effort")
		reasoning, _ := root.Lookup("reasoning")
		options, _ := root.Lookup("chat_template_kwargs")
		_, hasThinking := options.Lookup("enable_thinking")
		_, hasEffort := options.Lookup("reasoning_effort")
		text, _ := effort.Text()
		if (effort.Kind() == requestnorm.Null || (effort.Kind() == requestnorm.String && text == "")) &&
			!effortTruthy(reasoning) && !hasThinking && !hasEffort {
			req.ReasoningEffort, err = requestnorm.ParseValue([]byte(strconv.Quote(s.cfg.DefaultReasoningEffort)))
			if err != nil {
				return requestnorm.Result{}, o, err
			}
		}
	}
	for _, m := range req.Messages {
		for _, p := range m.Content.Parts {
			if p.Type == "image" && s.cfg.Vision == nil {
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
		result.Messages = addObjectDirective(result.Messages, false, o.schema)
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
func fitOutputTokens(contextLimit, promptTokens, requested int, strict bool, explicit bool) (int, error) {
	room := contextLimit - 8 - promptTokens
	if strict && explicit && requested > room {
		return 0, fmt.Errorf("request exceeds the context window: prompt (%d tokens) + max tokens (%d) exceeds the context (%d); requests are never truncated. Send a smaller max_tokens (at most %d here), or add \"fit_max_tokens\": true to the model's strata-<model>.json to shorten it to the room left (#545)", promptTokens, requested, contextLimit, max(0, room))
	}
	if room < 1 {
		return 0, fmt.Errorf("request exceeds the context window: prompt (%d tokens) leaves no room to answer in the context (%d); requests are never truncated", promptTokens, contextLimit)
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
	responseModel := s.modelFor(data)
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
	ids, embeddings, err := s.prepareImages(r, input.Messages, ids)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	if embeddings != "" {
		defer removeImageFile(embeddings)
	}
	ids, err = appendForcedPrompt(s.cfg.Tokenizer, rendered, ids, opt.force, s.cfg.MaxPromptBytes)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	opt.max, err = fitOutputTokens(s.cfg.Engine.ContextLimit(), len(ids), opt.max, s.cfg.DisableFitMaxTokens, opt.explicitMax)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	opt.budget = s.effectiveReasoningBudget(rendered.Thinking, opt.max, opt.budget)
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
	if opt.force != "" && !rendered.Thinking {
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
	var collector *logprobCollector
	var contentLogprobs, reasonLogprobs []logprobEntry
	var onLogprob func(int, *engine.Logprob) error
	if opt.sampling.Logprobs != nil {
		collector = newLogprobCollector(s.cfg.Tokenizer, s.stopIDs, opt.sampling.AllowedTokenIDs)
		onLogprob = func(id int, lp *engine.Logprob) error {
			if lp == nil {
				return nil
			}
			in := logprobInput{ID: id, Logprob: lp.Value}
			for _, alt := range lp.Alternatives {
				in.Alternatives = append(in.Alternatives, logprobAlternative{ID: alt.TokenID, Logprob: alt.Value})
			}
			return collector.Add(in)
		}
	}
	matcher := streamtext.NewStopMatcher(opt.stops)
	started := false
	structuredReady := false
	var timings map[string]any
	send := func(delta map[string]any, finish any, usage any, logprobs map[string]any) error {
		if !opt.stream || (opt.structured && !structuredReady) {
			return nil
		}
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			started = true
		}
		choice := map[string]any{"index": 0, "delta": delta, "finish_reason": finish}
		if len(logprobs) > 0 {
			choice["logprobs"] = logprobs
		}
		msg := map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": responseModel, "choices": []any{choice}}
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
		var lp map[string]any
		if collector != nil {
			entries := collector.Spell(text)
			if len(entries) > 0 {
				if !opt.stream || opt.structured {
					contentLogprobs = append(contentLogprobs, entries...)
				}
				lp = map[string]any{"content": entries}
			}
		}
		return send(map[string]any{"content": text}, nil, nil, lp)
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
					var lp map[string]any
					if collector != nil {
						entries := collector.Spell(ev.Text)
						if len(entries) > 0 {
							if !opt.stream || opt.structured {
								reasonLogprobs = append(reasonLogprobs, entries...)
							}
							lp = map[string]any{"reasoning_content": entries}
						}
					}
					if e := send(map[string]any{"reasoning_content": ev.Text}, nil, nil, lp); e != nil {
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
				if e = send(map[string]any{"tool_calls": []any{delta}}, nil, nil, nil); e != nil {
					return e
				}
			}
		}
		return nil
	}
	if opt.stream {
		if err = send(map[string]any{"role": "assistant", "content": ""}, nil, nil, nil); err != nil {
			return
		}
	}
	err = route(opening)
	var pumped generationResult
	if err == nil {
		force := ""
		if rendered.Thinking {
			force = opt.force
		}
		pumped, err = generateWithEmbeddings(r.Context(), s.cfg.Engine, s.cfg.Tokenizer, parser, ids, opt.max, opt.sampling, s.stopIDs, opt.budget, force, route, embeddings, onLogprob, streamHeartbeat(w, opt.stream, &started), s.repeatStop)
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
		validated, validationErr := validateStructuredJSON(content.String(), finish, opt.repair, opt.schema)
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
		validated, contentLogprobs = structuredLogprobs(content.String(), validated, contentLogprobs)
		if len(contentLogprobs) == 0 {
			reasonLogprobs = nil
		}
		content.Reset()
		content.WriteString(validated)
		structuredReady = true
		if opt.stream {
			if err = send(map[string]any{"role": "assistant", "content": ""}, nil, nil, nil); err != nil {
				return
			}
			delta := map[string]any{"content": validated}
			if reason.Len() > 0 {
				delta["reasoning_content"] = reason.String()
			}
			var lp map[string]any
			if len(contentLogprobs) > 0 {
				lp = map[string]any{"content": contentLogprobs}
				if len(reasonLogprobs) > 0 {
					lp["reasoning_content"] = reasonLogprobs
				}
			}
			if err = send(delta, nil, nil, lp); err != nil {
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
		if send(map[string]any{}, finish, usage, nil) == nil {
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
	choice := map[string]any{"index": 0, "message": msg, "finish_reason": finish}
	if len(contentLogprobs) > 0 || len(reasonLogprobs) > 0 {
		lp := map[string]any{"content": contentLogprobs, "refusal": nil}
		if contentLogprobs == nil {
			lp["content"] = []logprobEntry{}
		}
		if len(reasonLogprobs) > 0 {
			lp["reasoning_content"] = reasonLogprobs
		}
		choice["logprobs"] = lp
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"id": id, "object": "chat.completion", "created": created, "model": responseModel, "choices": []any{choice}, "usage": usage, "timings": timings})
}

func normalizeAliases(model string, aliases []string) []string {
	seen := map[string]bool{model: true}
	out := make([]string, 0, len(aliases))
	for _, alias := range aliases {
		alias = strings.TrimSpace(alias)
		if alias != "" && !seen[alias] {
			seen[alias] = true
			out = append(out, alias)
		}
	}
	return out
}

func (s *server) modelFor(data []byte) string {
	root, err := requestnorm.ParseValue(data)
	if err != nil {
		return s.cfg.Model
	}
	model, ok := root.Lookup("model")
	if !ok || model.Kind() != requestnorm.String {
		return s.cfg.Model
	}
	name, _ := model.Text()
	for _, alias := range s.cfg.Aliases {
		if name == alias {
			return name
		}
	}
	return s.cfg.Model
}
