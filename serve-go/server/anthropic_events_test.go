package server

import (
	"encoding/json"
	"strings"
	"testing"

	"strata/outputparser"
)

func TestAnthropicAdapterFeedsIncrementally(t *testing.T) {
	var adapter anthropicAdapter
	start, err := adapter.Start("msg-1", "model-x", 12)
	if err != nil || len(start) != 1 || start[0].name != "message_start" {
		t.Fatalf("Start = %#v, %v", start, err)
	}
	first, err := adapter.Feed([]outputparser.Event{{Kind: "reasoning", Text: "plan"}})
	if err != nil || len(first) != 2 || first[0].name != "content_block_start" || first[1].name != "content_block_delta" {
		t.Fatalf("first Feed = %#v, %v", first, err)
	}
	second, err := adapter.Feed([]outputparser.Event{
		{Kind: "content", Text: "answer"},
		{Kind: "tool_call", Call: &outputparser.ToolCall{ID: "call-1", Name: "lookup", Arguments: map[string]any{"q": "café"}}},
	})
	if err != nil {
		t.Fatalf("second Feed: %v", err)
	}
	if len(second) != 7 || second[0].name != "content_block_stop" || second[1].name != "content_block_start" {
		t.Fatalf("second Feed emitted wrong order: %#v", second)
	}
	var blockKinds []string
	for _, event := range append(first, second...) {
		if event.name == "content_block_start" {
			blockKinds = append(blockKinds, event.payload["content_block"].(map[string]any)["type"].(string))
		}
	}
	if len(blockKinds) != 3 || blockKinds[0] != "thinking" || blockKinds[1] != "text" || blockKinds[2] != "tool_use" {
		t.Fatalf("block order = %v", blockKinds)
	}
	raw := second[5].payload["delta"].(map[string]any)["partial_json"].(string)
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil || args["q"] != "café" {
		t.Fatalf("tool JSON %q parsed as %#v (err %v)", raw, args, err)
	}
	finish, err := adapter.Finish("stop", nil, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(finish) != 2 || finish[0].name != "message_delta" || finish[1].name != "message_stop" {
		t.Fatalf("Finish = %#v", finish)
	}
	if got := finish[0].payload["delta"].(map[string]any)["stop_reason"]; got != "tool_use" {
		t.Fatalf("stop reason = %v", got)
	}
	usage := finish[0].payload["usage"].(map[string]int)
	if usage["input_tokens"] != 1 || usage["cache_creation_input_tokens"] != 11 || usage["cache_read_input_tokens"] != 0 || usage["output_tokens"] != 3 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestAnthropicAdapterFinishReasons(t *testing.T) {
	tests := []struct {
		name     string
		finish   string
		sequence *string
		want     string
	}{
		{"natural", "stop", nil, "end_turn"},
		{"length", "length", nil, "max_tokens"},
		{"sequence", "stop", ptrString("END"), "stop_sequence"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var adapter anthropicAdapter
			if _, err := adapter.Start("m", "x", 0); err != nil {
				t.Fatal(err)
			}
			got, err := adapter.Finish(tt.finish, tt.sequence, 0)
			if err != nil {
				t.Fatal(err)
			}
			delta := got[0].payload["delta"].(map[string]any)
			if delta["stop_reason"] != tt.want {
				t.Fatalf("stop reason = %v, want %s", delta["stop_reason"], tt.want)
			}
		})
	}
}

func TestAnthropicAdapterRejectsUnmarshalableToolArguments(t *testing.T) {
	var adapter anthropicAdapter
	if _, err := adapter.Start("m", "x", 0); err != nil {
		t.Fatal(err)
	}
	_, err := adapter.Feed([]outputparser.Event{{
		Kind: "tool_call",
		Call: &outputparser.ToolCall{ID: "call-1", Name: "broken", Arguments: map[string]any{"bad": make(chan int)}},
	}})
	if err == nil || !strings.Contains(err.Error(), "encode tool arguments") {
		t.Fatalf("Feed error = %v, want propagated JSON marshal error", err)
	}
}

func ptrString(s string) *string { return &s }
