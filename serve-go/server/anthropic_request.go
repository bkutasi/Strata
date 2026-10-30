package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"strata/engine"
	"strata/requestnorm"
)

// anthropicOptions contains generation controls not used while normalizing a
// Messages prompt.
type anthropicOptions struct {
	stream   bool
	noTools  bool
	force    string
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
	o.stops, err = parseStops(root)
	if err != nil {
		return requestnorm.AnthropicResult{}, o, err
	}
	var toolChoice requestnorm.Value
	maxSeen := false
	for _, f := range root.Fields() {
		v := f.Value
		switch f.Name {
		case "system", "messages", "tools", "thinking", "output_config":
		case "metadata", "service_tier":
			// Python accepts these client metadata fields without changing generation.
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
			var n int
			if json.Unmarshal(b, &n) != nil || n > s.cfg.MaxOutputTokens {
				return requestnorm.AnthropicResult{}, o, errors.New("invalid max_tokens")
			}
			if n > 0 {
				o.max = n
			}
			maxSeen = true
		case "stream":
			var ok bool
			o.stream, ok = v.Bool()
			if !ok {
				return requestnorm.AnthropicResult{}, o, errors.New("stream must be a boolean")
			}
		case "stop", "stop_sequences":
		case "tool_choice":
			toolChoice = v
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
	kind, force := normalizeForceChoice(toolChoice, result.Tools, s.cfg.Family)
	o.noTools, o.force = kind == "none", force
	if o.noTools {
		result.Tools = nil
	}
	return result, o, nil
}
