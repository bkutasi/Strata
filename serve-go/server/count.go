package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"

	"strata/requestnorm"
)

func countSampling(name string, value requestnorm.Value, integer bool) error {
	if value.Kind() == requestnorm.Null {
		return nil
	}
	if integer {
		if _, ok := value.Integer(); ok {
			return nil
		}
	} else if value.Kind() == requestnorm.Number {
		return nil
	}
	if value.Kind() == requestnorm.String {
		text, _ := value.Text()
		text = strings.TrimSpace(text)
		n, err := strconv.ParseFloat(text, 64)
		if strings.HasPrefix(strings.ToLower(strings.TrimLeft(text, "+-")), "0x") {
			return fmt.Errorf("invalid sampling field %q", name)
		}
		if err == nil && !math.IsInf(n, 0) && !math.IsNaN(n) && (!integer || math.Trunc(n) == n) {
			return nil
		}
	}
	return fmt.Errorf("invalid sampling field %q", name)
}

func countStop(root requestnorm.Value) error {
	stop, ok := root.Lookup("stop")
	name := "stop"
	if !ok || stop.Kind() == requestnorm.Null {
		stop, ok = root.Lookup("stop_sequences")
		name = "stop_sequences"
	}
	if !ok || stop.Kind() == requestnorm.Null || name == "stop" && stop.Kind() == requestnorm.String {
		return nil
	}
	if stop.Kind() != requestnorm.Array || name == "stop" && len(stop.Items()) > 4 {
		return fmt.Errorf("invalid %s", name)
	}
	for _, item := range stop.Items() {
		if item.Kind() != requestnorm.String {
			return fmt.Errorf("invalid %s", name)
		}
	}
	return nil
}

func (s *server) decodeAnthropicCount(data []byte) (requestnorm.AnthropicResult, error) {
	root, err := requestnorm.ParseValue(data)
	if err != nil {
		return requestnorm.AnthropicResult{}, err
	}
	if root.Kind() != requestnorm.Object {
		return requestnorm.AnthropicResult{}, errors.New("request must be an object")
	}
	fields := map[string]json.RawMessage{}
	for _, field := range root.Fields() {
		switch field.Name {
		case "system", "messages", "tools", "thinking", "output_config", "reasoning_budget_tokens":
			value, err := field.Value.MarshalJSON()
			if err != nil {
				return requestnorm.AnthropicResult{}, err
			}
			fields[field.Name] = value
		case "temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty":
			if err := countSampling(field.Name, field.Value, false); err != nil {
				return requestnorm.AnthropicResult{}, err
			}
		case "top_k", "penalty_last_n", "seed":
			if err := countSampling(field.Name, field.Value, true); err != nil {
				return requestnorm.AnthropicResult{}, err
			}
		case "repeat_penalty":
			canonical, ok := root.Lookup("repetition_penalty")
			if (!ok || canonical.Kind() == requestnorm.Null) && countSampling(field.Name, field.Value, false) != nil {
				return requestnorm.AnthropicResult{}, fmt.Errorf("invalid sampling field %q", field.Name)
			}
		case "model", "max_tokens", "stream", "stop", "stop_sequences", "tool_choice", "repeat_last_n", "metadata", "service_tier":
			// These fields do not change the prompt counted by the Python endpoint.
		default:
			return requestnorm.AnthropicResult{}, fmt.Errorf("unsupported request field %q", field.Name)
		}
	}
	if err := countStop(root); err != nil {
		return requestnorm.AnthropicResult{}, err
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		return requestnorm.AnthropicResult{}, err
	}
	req, err := requestnorm.DecodeAnthropic(encoded)
	if err != nil {
		return requestnorm.AnthropicResult{}, err
	}
	result, err := s.normalizer.NormalizeAnthropic(req, true)
	if err != nil {
		return result, err
	}
	for _, message := range result.Messages {
		if message.Content.Parts != nil {
			return requestnorm.AnthropicResult{}, errors.New("images are not supported")
		}
	}
	return result, nil
}

func (s *server) countTokens(w http.ResponseWriter, r *http.Request) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", "invalid body")
		return
	}
	input, err := s.decodeAnthropicCount(data)
	if err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	renderer, ok := s.cfg.Renderer.(interface {
		RenderAnthropic(context.Context, requestnorm.AnthropicResult) (Rendered, error)
	})
	if !ok {
		fail(w, http.StatusNotImplemented, "unsupported_endpoint", "Anthropic renderer is not available")
		return
	}
	rendered, err := renderer.RenderAnthropic(r.Context(), input)
	if err != nil {
		fail(w, http.StatusBadGateway, "server_error", "renderer failed")
		return
	}
	ids, err := s.cfg.Tokenizer.EncodeWithBudget(rendered.Prompt, true, rendered.Plain, s.cfg.MaxPromptBytes)
	if err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", "invalid rendered prompt")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]int{"input_tokens": len(ids)})
}
