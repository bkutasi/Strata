package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"strata/outputparser"
	"strata/requestnorm"
	"strata/streamtext"
)

func (s *server) anthropic(w http.ResponseWriter, r *http.Request) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		fail(w, 400, "invalid_request_error", "invalid body")
		return
	}
	input, opt, err := s.decodeAnthropicRequest(data)
	if err != nil {
		fail(w, 400, "invalid_request_error", err.Error())
		return
	}
	renderer, ok := s.cfg.Renderer.(interface {
		RenderAnthropic(context.Context, requestnorm.AnthropicResult) (Rendered, error)
	})
	if !ok {
		fail(w, 501, "unsupported_endpoint", "Anthropic renderer is not available")
		return
	}
	rendered, err := renderer.RenderAnthropic(r.Context(), input)
	if err != nil {
		fail(w, 502, "server_error", "renderer failed")
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
	opt.budget = effectiveReasoningBudget(s.cfg.Family, rendered.Thinking, opt.max, opt.budget)
	var tools []map[string]any
	for _, value := range input.Tools {
		b, e := value.MarshalJSON()
		if e != nil {
			fail(w, 400, "invalid_request_error", "invalid tool")
			return
		}
		var tool map[string]any
		decoder := json.NewDecoder(strings.NewReader(string(b)))
		decoder.UseNumber()
		if e = decoder.Decode(&tool); e != nil {
			fail(w, 400, "invalid_request_error", "invalid tool")
			return
		}
		tools = append(tools, tool)
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
	var adapter anthropicAdapter
	var collector anthropicCollector
	started := false
	send := func(events []anthropicEvent) error {
		for _, event := range events {
			if !opt.stream {
				if err := collector.Consume(event); err != nil {
					return err
				}
				continue
			}
			b, err := json.Marshal(event.payload)
			if err != nil {
				return err
			}
			if !started {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				started = true
			}
			if _, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.name, b); err != nil {
				return err
			}
			if err = http.NewResponseController(w).Flush(); err != nil {
				return err
			}
		}
		return nil
	}
	begin, err := adapter.Start("msg_"+hex.EncodeToString(random[:]), s.cfg.Model, len(ids))
	if err == nil {
		err = send(begin)
	}
	if err != nil {
		return
	}
	matcher := streamtext.NewStopMatcher(opt.stops)
	feed := func(event outputparser.Event) error {
		events, err := adapter.Feed([]outputparser.Event{event})
		if err != nil {
			return err
		}
		return send(events)
	}
	route := func(events []outputparser.Event) error {
		for _, event := range events {
			if event.Kind == "content" {
				event.Text = matcher.Push(event.Text)
				if err := feed(event); err != nil {
					return err
				}
				if matcher.Hit() != "" {
					return stopGeneration
				}
				continue
			}
			if text := matcher.Boundary(); text != "" {
				if err := feed(outputparser.Event{Kind: "content", Text: text}); err != nil {
					return err
				}
			}
			if err := feed(event); err != nil {
				return err
			}
		}
		return nil
	}
	err = route(opening)
	var result generationResult
	if err == nil {
		force := ""
		if rendered.Thinking {
			force = opt.force
		}
		result, err = generateWithForce(r.Context(), s.cfg.Engine, s.cfg.Tokenizer, parser, ids, opt.max, opt.sampling, s.stopIDs, opt.budget, force, route)
	}
	if err == nil {
		if text := matcher.Finish(); text != "" {
			err = feed(outputparser.Event{Kind: "content", Text: text})
		}
	}
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		if !started {
			fail(w, 503, "server_error", "generation failed; restart server if engine exited")
		} else {
			payload, _ := json.Marshal(map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": "generation failed"}})
			_, _ = fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
			_ = http.NewResponseController(w).Flush()
		}
		return
	}
	finish := result.result.Finish
	if result.stopped {
		finish = "stop"
	}
	var stop *string
	if hit := matcher.Hit(); hit != "" {
		stop = &hit
	}
	adapter.cacheRead = result.cached
	events, err := adapter.Finish(finish, stop, result.count)
	if err == nil {
		err = send(events)
	}
	if err != nil {
		if !started {
			fail(w, 500, "server_error", "could not encode response")
		}
		return
	}
	if !opt.stream {
		response, err := collector.Result()
		if err != nil {
			fail(w, 500, "server_error", "could not collect response")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}
}
