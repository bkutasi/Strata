package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"strata/engine"
	"strata/requestnorm"
)

// anthropicOptions contains generation controls not used while normalizing a
// Messages prompt.
type anthropicOptions struct {
	stream   bool
	noTools  bool
	max      int
	stops    []string
	sampling engine.Sampling
}

// decodeAnthropicRequest strictly decodes a supported Messages generation
// request. Prompt fields are delegated to the count endpoint's decoder so both
// endpoints produce exactly the same normalized AnthropicResult.
func (s *server) decodeAnthropicRequest(data []byte) (requestnorm.AnthropicResult, anthropicOptions, error) {
	root, err := requestnorm.ParseValue(data)
	if err != nil {
		return requestnorm.AnthropicResult{}, anthropicOptions{}, err
	}
	if root.Kind() != requestnorm.Object {
		return requestnorm.AnthropicResult{}, anthropicOptions{}, errors.New("request must be an object")
	}
	o := anthropicOptions{max: s.cfg.MaxOutputTokens}
	maxSeen := false
	for _, f := range root.Fields() {
		v := f.Value
		switch f.Name {
		case "system", "messages", "tools", "thinking", "output_config":
		case "model":
			m, ok := v.Text()
			if !ok || v.Kind() != requestnorm.String || m != s.cfg.Model {
				return requestnorm.AnthropicResult{}, o, errors.New("model not found")
			}
		case "max_tokens":
			if maxSeen {
				return requestnorm.AnthropicResult{}, o, errors.New("invalid max_tokens")
			}
			b, _ := v.MarshalJSON()
			if json.Unmarshal(b, &o.max) != nil || o.max < 1 || o.max > s.cfg.MaxOutputTokens {
				return requestnorm.AnthropicResult{}, o, errors.New("invalid max_tokens")
			}
			maxSeen = true
		case "stream":
			var ok bool
			o.stream, ok = v.Bool()
			if !ok {
				return requestnorm.AnthropicResult{}, o, errors.New("stream must be a boolean")
			}
		case "stop", "stop_sequences":
			// Python's stop takes precedence whenever it is non-null.
			if f.Name == "stop_sequences" {
				if stop, ok := root.Lookup("stop"); ok && stop.Kind() != requestnorm.Null {
					continue
				}
				if v.Kind() == requestnorm.Null {
					continue
				}
				if v.Kind() != requestnorm.Array {
					return requestnorm.AnthropicResult{}, o, errors.New("invalid stop_sequences")
				}
			}
			if v.Kind() == requestnorm.Null {
				continue
			}
			if f.Name == "stop" && v.Kind() == requestnorm.String {
				t, _ := v.Text()
				o.stops = []string{t}
			} else if v.Kind() == requestnorm.Array {
				o.stops = nil
				for _, item := range v.Items() {
					t, ok := item.Text()
					if !ok || item.Kind() != requestnorm.String {
						return requestnorm.AnthropicResult{}, o, fmt.Errorf("invalid %s", f.Name)
					}
					o.stops = append(o.stops, t)
				}
			} else {
				return requestnorm.AnthropicResult{}, o, fmt.Errorf("invalid %s", f.Name)
			}
			if f.Name == "stop" && len(o.stops) > 4 {
				return requestnorm.AnthropicResult{}, o, errors.New("at most four stop strings")
			}
			for _, t := range o.stops {
				if !utf8.ValidString(t) || len(t) > 1024 {
					return requestnorm.AnthropicResult{}, o, errors.New("invalid stop string")
				}
			}
		case "tool_choice":
			o.noTools, err = toolChoiceNone(v)
			if err != nil {
				return requestnorm.AnthropicResult{}, o, err
			}
		case "reasoning_budget_tokens":
			// The prompt normalizer accepts this for count parity, but generation
			// does not yet apply the budget.
			if v.Kind() != requestnorm.Null {
				return requestnorm.AnthropicResult{}, o, errors.New("reasoning budget is not supported")
			}
		case "temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty",
			"penalty_last_n", "seed", "repeat_penalty", "repeat_last_n":
			if err := setSampling(&o.sampling, root, f.Name, v); err != nil {
				return requestnorm.AnthropicResult{}, o, err
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
					return requestnorm.AnthropicResult{}, o, errors.New("top_k must be an integer")
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
				return requestnorm.AnthropicResult{}, o, errors.New("top_k must be an integer")
			}
			o.sampling.TopK = &k
		default:
			return requestnorm.AnthropicResult{}, o, fmt.Errorf("unsupported request field %q", f.Name)
		}
	}
	// The count decoder validates prompt field shapes and normalization; it
	// cannot be passed an extracted object because it validates sampling too.
	result, err := s.decodeAnthropicCount(data)
	if err != nil {
		return result, o, err
	}
	if result.ReasoningBudgetTokens.Kind() != requestnorm.Null {
		return requestnorm.AnthropicResult{}, o, errors.New("reasoning budget is not supported")
	}
	if err = requestnorm.RequireAnthropicMessages(result.Messages); err != nil {
		return result, o, err
	}
	for _, m := range result.Messages {
		if m.Content.Parts != nil {
			return requestnorm.AnthropicResult{}, o, errors.New("images are not supported")
		}
	}
	if o.noTools {
		result.Tools = nil
	}
	return result, o, nil
}
