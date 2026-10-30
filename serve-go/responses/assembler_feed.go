package responses

import (
	"errors"

	"strata/requestnorm"
)

// Feed adds one parser event to the response and returns the resulting events.
func (a *Assembler) Feed(ev ParserEvent) ([]Event, error) {
	if err := a.ready(); err != nil {
		return nil, err
	}
	var out []Event
	switch ev.Kind {
	case "reasoning":
		if ev.Text == "" {
			return nil, nil
		}
		if a.item == nil || a.item["type"] != "reasoning" {
			out = append(out, a.close("")...)
			id, err := a.makeID("rs")
			if err != nil {
				return out, err
			}
			out = append(out, a.open(map[string]any{
				"id": id, "type": "reasoning", "status": "in_progress", "summary": []any{},
				"content": []any{map[string]any{"type": "reasoning_text", "text": ""}},
			})...)
		}
		part := a.item["content"].([]any)[0].(map[string]any)
		part["text"] = part["text"].(string) + ev.Text
		out = append(out, a.event("response.reasoning_text.delta", map[string]any{
			"item_id": a.item["id"], "output_index": a.index(), "content_index": 0, "delta": ev.Text,
		}))
	case "content":
		if ev.Text == "" {
			return nil, nil
		}
		if a.jsonMode {
			out = append(out, a.close("")...)
			a.jsonText.WriteString(ev.Text)
			return out, nil
		}
		if a.item == nil || a.item["type"] != "message" {
			out = append(out, a.close("")...)
			id, err := a.makeID("msg")
			if err != nil {
				return out, err
			}
			out = append(out, a.open(map[string]any{
				"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{},
			})...)
		}
		part := a.item["content"].([]any)[0].(map[string]any)
		part["text"] = part["text"].(string) + ev.Text
		out = append(out, a.event("response.output_text.delta", map[string]any{
			"item_id": a.item["id"], "output_index": a.index(), "content_index": 0,
			"delta": ev.Text, "logprobs": []any{},
		}))
	case "tool_start":
		if ev.Call == nil {
			return out, errors.New("responses: tool_start event has no call")
		}
		out = append(out, a.close("")...)
		var err error
		out, err = appendCallStarted(a, out)
		if err != nil {
			return out, err
		}
		item, err := a.callItem(ev.Call)
		if err != nil {
			return out, err
		}
		out = append(out, a.open(item)...)
	case "tool_args":
		if ev.Call == nil || a.item == nil || a.item["call_id"] != ev.Call.ID {
			return out, nil
		}
		if a.item["type"] == "function_call" {
			a.item["arguments"] = a.item["arguments"].(string) + ev.Text
			out = append(out, a.event("response.function_call_arguments.delta", map[string]any{
				"item_id": a.item["id"], "output_index": a.index(), "delta": ev.Text,
			}))
		}
	case "tool_call":
		if ev.Call == nil {
			return out, errors.New("responses: tool_call event has no call")
		}
		item := a.item
		if item == nil || item["call_id"] != ev.Call.ID {
			out = append(out, a.close("")...)
			var err error
			out, err = appendCallStarted(a, out)
			if err != nil {
				return out, err
			}
			created, err := a.callItem(ev.Call)
			if err != nil {
				return out, err
			}
			out = append(out, a.open(created)...)
			item = a.item
			if item["type"] == "function_call" {
				arguments, err := pythonJSON(ev.Call.Arguments)
				if err != nil {
					return out, err
				}
				item["arguments"] = arguments
				out = append(out, a.event("response.function_call_arguments.delta", map[string]any{
					"item_id": item["id"], "output_index": a.index(), "delta": arguments,
				}))
			}
		}
		if item["type"] == "custom_tool_call" {
			var input string
			if ev.Call.Arguments.Kind() == requestnorm.Object {
				if value, ok := ev.Call.Arguments.Lookup("input"); ok {
					if value.Kind() == requestnorm.String {
						input, _ = stringOf(value)
					} else {
						var err error
						input, err = pythonJSON(ev.Call.Arguments)
						if err != nil {
							return out, err
						}
					}
				} else {
					var err error
					input, err = pythonJSON(ev.Call.Arguments)
					if err != nil {
						return out, err
					}
				}
			} else {
				var err error
				input, err = pythonJSON(ev.Call.Arguments)
				if err != nil {
					return out, err
				}
			}
			item["input"] = input
			out = append(out, a.event("response.custom_tool_call_input.delta", map[string]any{
				"item_id": item["id"], "output_index": a.index(), "delta": input,
			}))
		}
		item["status"] = "completed"
		out = append(out, a.close("completed")...)
	}
	return out, nil
}

func appendCallStarted(a *Assembler, out []Event) ([]Event, error) {
	events, err := a.callStarted()
	if err != nil {
		return out, err
	}
	return append(out, events...), nil
}
