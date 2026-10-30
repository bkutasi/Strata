package server

import (
	"encoding/json"
	"fmt"

	"strata/outputparser"
)

// anthropicEvent is one named Server-Sent Event and its JSON payload.
type anthropicEvent struct {
	name    string
	payload map[string]any
}

// anthropicAdapter converts one generation to Anthropic Messages events.
// Feed returns events as they become available; only the current block state
// is retained between calls.
type anthropicAdapter struct {
	started     bool
	finished    bool
	index       int
	open        string
	usedTool    bool
	inputTokens int
	id          string
	model       string
}

func (a *anthropicAdapter) Start(id, model string, inputTokens int) ([]anthropicEvent, error) {
	if a.started {
		return nil, fmt.Errorf("anthropic events: generation already started")
	}
	a.started = true
	a.index = -1
	a.id, a.model, a.inputTokens = id, model, inputTokens
	message := map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": model,
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]int{"input_tokens": inputTokens, "output_tokens": 0},
	}
	return []anthropicEvent{{"message_start", map[string]any{"type": "message_start", "message": message}}}, nil
}

func (a *anthropicAdapter) Feed(events []outputparser.Event) ([]anthropicEvent, error) {
	if !a.started || a.finished {
		return nil, fmt.Errorf("anthropic events: Feed outside active generation")
	}
	var out []anthropicEvent
	closeBlock := func() {
		if a.open != "" {
			out = append(out, anthropicEvent{"content_block_stop", map[string]any{"type": "content_block_stop", "index": a.index}})
			a.open = ""
		}
	}
	startBlock := func(kind string, block map[string]any) {
		a.index++
		a.open = kind
		out = append(out, anthropicEvent{"content_block_start", map[string]any{"type": "content_block_start", "index": a.index, "content_block": block}})
	}
	for _, ev := range events {
		switch ev.Kind {
		case "reasoning", "content":
			if ev.Text == "" {
				continue
			}
			kind, deltaType, field := "text", "text_delta", "text"
			if ev.Kind == "reasoning" {
				kind, deltaType, field = "thinking", "thinking_delta", "thinking"
			}
			if a.open != kind {
				closeBlock()
				block := map[string]any{"type": kind, field: ""}
				if kind == "thinking" {
					block["signature"] = ""
				}
				startBlock(kind, block)
			}
			out = append(out, anthropicEvent{"content_block_delta", map[string]any{"type": "content_block_delta", "index": a.index, "delta": map[string]any{"type": deltaType, field: ev.Text}}})
		case "tool_call":
			if ev.Call == nil {
				continue
			}
			raw, err := json.Marshal(ev.Call.Arguments)
			if err != nil {
				return out, fmt.Errorf("anthropic events: encode tool arguments: %w", err)
			}
			closeBlock()
			a.usedTool = true
			startBlock("tool_use", map[string]any{"type": "tool_use", "id": ev.Call.ID, "name": ev.Call.Name, "input": map[string]any{}})
			out = append(out, anthropicEvent{"content_block_delta", map[string]any{"type": "content_block_delta", "index": a.index, "delta": map[string]any{"type": "input_json_delta", "partial_json": string(raw)}}})
			closeBlock()
		}
	}
	return out, nil
}

func (a *anthropicAdapter) Finish(finish string, stopSequence *string, completionTokens int) ([]anthropicEvent, error) {
	if !a.started || a.finished {
		return nil, fmt.Errorf("anthropic events: Finish outside active generation")
	}
	a.finished = true
	var out []anthropicEvent
	if a.open != "" {
		out = append(out, anthropicEvent{"content_block_stop", map[string]any{"type": "content_block_stop", "index": a.index}})
		a.open = ""
	}
	stopReason := "end_turn"
	if stopSequence != nil {
		stopReason = "stop_sequence"
	} else if finish == "length" {
		stopReason = "max_tokens"
	} else if finish == "stop" && a.usedTool {
		stopReason = "tool_use"
	}
	inputRead := a.inputTokens
	if inputRead > 0 {
		inputRead = 1
	}
	out = append(out, anthropicEvent{"message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": stopSequence},
		"usage": map[string]int{"input_tokens": inputRead, "cache_creation_input_tokens": a.inputTokens - inputRead, "cache_read_input_tokens": 0, "output_tokens": completionTokens},
	}})
	out = append(out, anthropicEvent{"message_stop", map[string]any{"type": "message_stop"}})
	return out, nil
}
