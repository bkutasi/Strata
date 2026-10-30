package server

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// anthropicCollector folds named events into one non-streaming response.
// It retains only the response and the current tool-input JSON fragment.
type anthropicCollector struct {
	message map[string]any
	blocks  []any
	current map[string]any
	rawJSON strings.Builder
	hasJSON bool
}

func (c *anthropicCollector) Consume(event anthropicEvent) error {
	e := event.payload
	switch event.name {
	case "message_start":
		if c.message != nil {
			return fmt.Errorf("anthropic collect: duplicate message_start")
		}
		m, ok := e["message"].(map[string]any)
		if !ok {
			return fmt.Errorf("anthropic collect: message_start missing message object")
		}
		c.message = cloneAnthropicMap(m)
		switch usage := c.message["usage"].(type) {
		case map[string]int:
			normalized := make(map[string]any, len(usage))
			for k, v := range usage {
				normalized[k] = v
			}
			c.message["usage"] = normalized
		case map[string]any:
		default:
			return fmt.Errorf("anthropic collect: message usage is not an object")
		}
	case "content_block_start":
		b, ok := e["content_block"].(map[string]any)
		if !ok {
			return fmt.Errorf("anthropic collect: content_block_start missing content_block object")
		}
		if c.current != nil {
			return fmt.Errorf("anthropic collect: content block started before previous block stopped")
		}
		c.current = cloneAnthropicMap(b)
		c.blocks = append(c.blocks, c.current)
		c.rawJSON.Reset()
		c.hasJSON = false
	case "content_block_delta":
		if c.current == nil {
			return fmt.Errorf("anthropic collect: delta without active content block")
		}
		d, ok := e["delta"].(map[string]any)
		if !ok {
			return fmt.Errorf("anthropic collect: content_block_delta missing delta object")
		}
		switch d["type"] {
		case "text_delta":
			s, ok := d["text"].(string)
			if !ok {
				return fmt.Errorf("anthropic collect: text_delta missing text")
			}
			v, _ := c.current["text"].(string)
			c.current["text"] = v + s
		case "thinking_delta":
			s, ok := d["thinking"].(string)
			if !ok {
				return fmt.Errorf("anthropic collect: thinking_delta missing thinking")
			}
			v, _ := c.current["thinking"].(string)
			c.current["thinking"] = v + s
		case "input_json_delta":
			s, ok := d["partial_json"].(string)
			if !ok {
				return fmt.Errorf("anthropic collect: input_json_delta missing partial_json")
			}
			c.rawJSON.WriteString(s)
			c.hasJSON = true
		default:
			return fmt.Errorf("anthropic collect: unknown content delta type %v", d["type"])
		}
	case "content_block_stop":
		if c.current == nil {
			return fmt.Errorf("anthropic collect: content_block_stop without active block")
		}
		if c.hasJSON {
			decoder := json.NewDecoder(strings.NewReader(c.rawJSON.String()))
			decoder.UseNumber()
			var input any
			if err := decoder.Decode(&input); err != nil {
				c.blocks = c.blocks[:len(c.blocks)-1]
			} else {
				var trailing any
				if err := decoder.Decode(&trailing); err != io.EOF {
					c.blocks = c.blocks[:len(c.blocks)-1]
				} else {
					c.current["input"] = input
				}
			}
		}
		c.current = nil
		c.rawJSON.Reset()
		c.hasJSON = false
	case "message_delta":
		if c.message == nil {
			return fmt.Errorf("anthropic collect: message_delta before message_start")
		}
		d, ok := e["delta"].(map[string]any)
		if !ok {
			return fmt.Errorf("anthropic collect: message_delta missing delta object")
		}
		if _, ok := d["stop_reason"]; !ok {
			return fmt.Errorf("anthropic collect: message_delta missing stop_reason")
		}
		c.message["stop_reason"] = d["stop_reason"]
		c.message["stop_sequence"] = d["stop_sequence"]
		u, ok := c.message["usage"].(map[string]any)
		if !ok {
			return fmt.Errorf("anthropic collect: message usage is not an object")
		}
		switch usage := e["usage"].(type) {
		case map[string]any:
			for k, v := range usage {
				u[k] = v
			}
		case map[string]int:
			for k, v := range usage {
				u[k] = v
			}
		default:
			return fmt.Errorf("anthropic collect: message_delta missing usage object")
		}
		if rejected, ok := e["strata_rejected_tool_calls"]; ok && rejected != nil {
			c.message["strata_rejected_tool_calls"] = rejected
		}
	case "message_stop":
		// End marker has no additional response fields.
	default:
		return fmt.Errorf("anthropic collect: unknown event %q", event.name)
	}
	return nil
}

func (c *anthropicCollector) Result() (map[string]any, error) {
	if c.message == nil {
		return nil, fmt.Errorf("anthropic collect: missing message_start")
	}
	if c.current != nil {
		return nil, fmt.Errorf("anthropic collect: unterminated content block")
	}
	if c.blocks == nil {
		c.blocks = []any{}
	}
	for i := 0; i+1 < len(c.blocks); i++ {
		a, aok := c.blocks[i].(map[string]any)
		b, bok := c.blocks[i+1].(map[string]any)
		if aok && bok && a["type"] == "thinking" && b["type"] == "text" {
			thinking, _ := a["thinking"].(string)
			text, _ := b["text"].(string)
			if strings.TrimSpace(thinking) == strings.TrimSpace(text) {
				c.blocks = append(c.blocks[:i], c.blocks[i+1:]...)
				break
			}
		}
	}
	c.message["content"] = c.blocks
	return c.message, nil
}

func cloneAnthropicMap(src map[string]any) map[string]any {
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}
