package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"strata/outputparser"
	"strata/requestnorm"
	"strata/responses"
)

func failResponsesRequest(w http.ResponseWriter, err error) {
	var responseErr *responses.Error
	if !errors.As(err, &responseErr) {
		fail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	status := responseErr.Status
	if status == 0 {
		status = http.StatusBadRequest
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(responseErr.Body())
}

func (s *server) responses(w http.ResponseWriter, r *http.Request) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		fail(w, 400, "invalid_request_error", "invalid body")
		return
	}
	req, err := responses.Decode(data)
	if err != nil {
		failResponsesRequest(w, err)
		return
	}
	prepared, err := responses.Prepare(req, responses.PrepareOptions{Family: s.cfg.Family})
	if err != nil {
		failResponsesRequest(w, err)
		return
	}
	root, err := requestnorm.ParseValue(data)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	o := options{max: s.cfg.MaxOutputTokens}
	for _, f := range root.Fields() {
		v := f.Value
		switch f.Name {
		case "model":
			if v.Kind() != requestnorm.Null {
				m, ok := v.Text()
				if !ok || m != s.cfg.Model {
					err = errors.New("model not found")
				}
			}
		case "stream":
			o.stream, _ = v.Bool()
		case "max_output_tokens":
			if v.Kind() != requestnorm.Null {
				text, ok := v.Integer()
				n, e := strconv.Atoi(text)
				if !ok || e != nil || n > s.cfg.MaxOutputTokens {
					err = errors.New("invalid max_output_tokens")
				} else if n > 0 {
					o.max = n
				}
			}
		case "temperature", "top_p":
			err = setSampling(&o.sampling, root, f.Name, v)
		}
		if err != nil {
			fail(w, 400, "invalid_request_error", err.Error())
			return
		}
	}
	if prepared.Format.Kind() != requestnorm.Null {
		fail(w, 400, "invalid_request_error", "structured output is not supported")
		return
	}
	for _, name := range prepared.Names {
		if name.Kind != "function" {
			fail(w, 400, "invalid_request_error", "only function tools are supported")
			return
		}
	}
	for _, m := range prepared.Messages {
		if m.Content.Parts != nil {
			fail(w, 400, "invalid_request_error", "images are not supported")
			return
		}
	}
	if err = requestnorm.RequireMessages(prepared.Messages); err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	input := requestnorm.Result{Messages: prepared.Messages, Tools: prepared.Tools, Kwargs: prepared.Kwargs}
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
	o.max, err = fitOutputTokens(s.cfg.Engine.ContextLimit(), len(ids), o.max)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	var tools []map[string]any
	for _, v := range prepared.Tools {
		b, e := v.MarshalJSON()
		if e != nil {
			fail(w, 400, "invalid_request_error", "invalid tool")
			return
		}
		var t map[string]any
		dec := json.NewDecoder(strings.NewReader(string(b)))
		dec.UseNumber()
		if dec.Decode(&t) != nil {
			fail(w, 400, "invalid_request_error", "invalid tool")
			return
		}
		tools = append(tools, t)
	}
	opts := outputparser.Options{Thinking: rendered.Thinking, Tools: tools, DeclaredOnly: true}
	var parser generationParser
	if s.cfg.Family == requestnorm.FamilyDeepSeek {
		parser, err = outputparser.NewDeepSeek(opts)
	} else {
		parser, err = outputparser.New(opts)
	}
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	a, err := responses.NewAssembler(req, prepared, responses.AssemblerOptions{Model: s.cfg.Model, PromptTokens: int64(len(ids))})
	if err != nil {
		fail(w, 500, "server_error", err.Error())
		return
	}
	started := false
	var final any
	send := func(events []responses.Event) error {
		for _, event := range events {
			if response, ok := event["response"]; ok && (event["type"] == "response.completed" || event["type"] == "response.incomplete") {
				final = response
				if timings, ok := event["timings"]; ok && !o.stream {
					response, ok := response.(map[string]any)
					if !ok {
						return errors.New("invalid final response")
					}
					response["timings"] = timings
				}
			}
			if !o.stream {
				continue
			}
			b, e := json.Marshal(event)
			if e != nil {
				return e
			}
			if !started {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				started = true
			}
			if _, e = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event["type"], b); e != nil {
				return e
			}
			if e = http.NewResponseController(w).Flush(); e != nil {
				return e
			}
		}
		return nil
	}
	events, err := a.Start()
	if err == nil {
		err = send(events)
	}
	feed := func(ev responses.ParserEvent) error {
		events, e := a.Feed(ev)
		if e != nil {
			return e
		}
		return send(events)
	}
	route := func(events []outputparser.Event) error {
		for _, event := range events {
			switch event.Kind {
			case "content", "reasoning":
				if e := feed(responses.ParserEvent{Kind: event.Kind, Text: event.Text}); e != nil {
					return e
				}
			case "tool_call":
				// The final parser value preserves arbitrary-size integers as json.Number.
				// Marshal rejects non-finite values, rather than silently changing them.
				b, e := responseArguments(event.Call.Arguments)
				if e != nil {
					return e
				}
				value, e := requestnorm.ParseValue(b)
				if e != nil {
					return e
				}
				call := &responses.Call{ID: event.Call.ID, Name: event.Call.Name, Arguments: value}
				// Supplying the serialized final arguments as a delta also avoids the
				// assembler's Python-format float restriction. No provisional call is
				// emitted before the parser confirms completion.
				for _, part := range []responses.ParserEvent{{Kind: "tool_start", Call: call}, {Kind: "tool_args", Text: string(b), Call: call}, {Kind: "tool_call", Call: call}} {
					if e = feed(part); e != nil {
						return e
					}
				}
			}
		}
		return nil
	}
	var result generationResult
	if err == nil {
		result, err = generate(r.Context(), s.cfg.Engine, s.cfg.Tokenizer, parser, ids, o.max, o.sampling, s.stopIDs, route)
	}
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		if !started {
			fail(w, 503, "server_error", "generation failed")
		} else if event, e := a.Failed("generation failed", "server_error"); e == nil {
			_ = send([]responses.Event{event})
		}
		return
	}
	finish := result.result.Finish
	if result.stopped || finish == "cancel" {
		finish = "stop"
	}
	events, err = a.Finish(responses.Done{Finish: finish, CompletionTokens: int64(result.count), Reused: int64(result.cached), Timings: result.timings})
	if err == nil {
		err = send(events)
	}
	if err != nil {
		if !started {
			fail(w, 500, "server_error", "could not encode response")
		}
		return
	}
	if !o.stream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(final)
	}
}

// responseArguments matches Python's float repr and preserves json.Number
// integers. Non-finite floats cannot be represented in a Responses JSON value.
func responseArguments(v any) ([]byte, error) {
	switch x := v.(type) {
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, errors.New("non-finite tool argument")
		}
		abs := math.Abs(x)
		format := byte('f')
		if abs != 0 && (abs < 1e-4 || abs >= 1e16) {
			format = 'e'
		}
		text := strconv.FormatFloat(x, format, -1, 64)
		if format == 'f' && !strings.Contains(text, ".") {
			text += ".0"
		}
		return []byte(text), nil
	case map[string]any:
		keys := make([]string, 0, len(x))
		for key := range x {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		var b strings.Builder
		b.WriteByte('{')
		for i, key := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			k, _ := json.Marshal(key)
			value, err := responseArguments(x[key])
			if err != nil {
				return nil, err
			}
			b.Write(k)
			b.WriteString(": ")
			b.Write(value)
		}
		b.WriteByte('}')
		return []byte(b.String()), nil
	case []any:
		var b strings.Builder
		b.WriteByte('[')
		for i, value := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			part, err := responseArguments(value)
			if err != nil {
				return nil, err
			}
			b.Write(part)
		}
		b.WriteByte(']')
		return []byte(b.String()), nil
	default:
		return json.Marshal(v)
	}
}
