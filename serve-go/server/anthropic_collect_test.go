package server

import (
	"encoding/json"
	"reflect"
	"testing"

	"strata/outputparser"
)

func collectFixture(t *testing.T, events ...anthropicEvent) map[string]any {
	t.Helper()
	c := &anthropicCollector{}
	for _, e := range events {
		if err := c.Consume(e); err != nil {
			t.Fatalf("Consume(%q): %v", e.name, err)
		}
	}
	got, err := c.Result()
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func started() anthropicEvent {
	return anthropicEvent{"message_start", map[string]any{"message": map[string]any{
		"id": "m1", "type": "message", "role": "assistant", "model": "test",
		"content": []any{}, "usage": map[string]any{"input_tokens": 3, "output_tokens": 0},
	}}}
}

func blockStart(kind string, fields map[string]any) anthropicEvent {
	fields["type"] = kind
	return anthropicEvent{"content_block_start", map[string]any{"content_block": fields}}
}

func delta(kind string, fields map[string]any) anthropicEvent {
	fields["type"] = kind
	return anthropicEvent{"content_block_delta", map[string]any{"delta": fields}}
}

func TestAnthropicCollectorText(t *testing.T) {
	got := collectFixture(t, started(), blockStart("text", map[string]any{"text": ""}),
		delta("text_delta", map[string]any{"text": "hello"}), anthropicEvent{"content_block_stop", map[string]any{}})
	if content := got["content"].([]any); content[0].(map[string]any)["text"] != "hello" {
		t.Fatalf("content = %#v", content)
	}
}

func TestAnthropicCollectorReasoning(t *testing.T) {
	got := collectFixture(t, started(), blockStart("thinking", map[string]any{"thinking": "", "signature": ""}),
		delta("thinking_delta", map[string]any{"thinking": "reason"}), anthropicEvent{"content_block_stop", map[string]any{}})
	if content := got["content"].([]any); content[0].(map[string]any)["thinking"] != "reason" {
		t.Fatalf("content = %#v", content)
	}
}

func TestAnthropicCollectorToolInputDeltas(t *testing.T) {
	got := collectFixture(t, started(), blockStart("tool_use", map[string]any{"id": "t1", "name": "f", "input": map[string]any{}}),
		delta("input_json_delta", map[string]any{"partial_json": `{"x":`}),
		delta("input_json_delta", map[string]any{"partial_json": `1,"y":[true]}`}),
		anthropicEvent{"content_block_stop", map[string]any{}})
	input := got["content"].([]any)[0].(map[string]any)["input"]
	want := map[string]any{"x": json.Number("1"), "y": []any{true}}
	if !reflect.DeepEqual(input, want) {
		t.Fatalf("input = %#v, want %#v", input, want)
	}
}

func TestAnthropicCollectorDropsIncompleteTool(t *testing.T) {
	got := collectFixture(t, started(), blockStart("tool_use", map[string]any{"id": "t1", "name": "f", "input": map[string]any{}}),
		delta("input_json_delta", map[string]any{"partial_json": `{"x":`}),
		anthropicEvent{"content_block_stop", map[string]any{}})
	if content := got["content"].([]any); len(content) != 0 {
		t.Fatalf("content = %#v, want incomplete tool discarded", content)
	}
}

func TestAnthropicCollectorDropsTrailingToolGarbage(t *testing.T) {
	got := collectFixture(t, started(), blockStart("tool_use", map[string]any{"id": "t1", "name": "f", "input": map[string]any{}}),
		delta("input_json_delta", map[string]any{"partial_json": `{"x":1} garbage`}),
		anthropicEvent{"content_block_stop", map[string]any{}})
	if content := got["content"].([]any); len(content) != 0 {
		t.Fatalf("content = %#v, want malformed tool discarded", content)
	}
}

func TestAnthropicCollectorAllowsWhitespaceAfterToolJSON(t *testing.T) {
	got := collectFixture(t, started(), blockStart("tool_use", map[string]any{"id": "t1", "name": "f", "input": map[string]any{}}),
		delta("input_json_delta", map[string]any{"partial_json": "{\"x\":9007199254740993} \t\n "}),
		anthropicEvent{"content_block_stop", map[string]any{}})
	input := got["content"].([]any)[0].(map[string]any)["input"].(map[string]any)
	if input["x"] != json.Number("9007199254740993") {
		t.Fatalf("input = %#v, want exact JSON integer", input)
	}
}

func TestAnthropicCollectorIdenticalThinkingFallback(t *testing.T) {
	got := collectFixture(t, started(),
		blockStart("thinking", map[string]any{"thinking": "", "signature": ""}),
		delta("thinking_delta", map[string]any{"thinking": "answer"}),
		anthropicEvent{"content_block_stop", map[string]any{}},
		blockStart("text", map[string]any{"text": ""}),
		delta("text_delta", map[string]any{"text": " answer "}),
		anthropicEvent{"content_block_stop", map[string]any{}})
	content := got["content"].([]any)
	if len(content) != 1 || content[0].(map[string]any)["type"] != "text" {
		t.Fatalf("content = %#v, want only text block", content)
	}
}

func TestAnthropicCollectorMalformedStructureReturnsError(t *testing.T) {
	c := &anthropicCollector{}
	if err := c.Consume(anthropicEvent{"content_block_delta", map[string]any{}}); err == nil {
		t.Fatal("expected error for delta without block")
	}
}

func TestAnthropicCollectorAdapterTextEndToEnd(t *testing.T) {
	var adapter anthropicAdapter
	events, err := adapter.Start("m1", "test", 7)
	if err != nil {
		t.Fatal(err)
	}
	feed, err := adapter.Feed([]outputparser.Event{{Kind: "content", Text: "hello"}})
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, feed...)
	finish, err := adapter.Finish("stop", nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, finish...)
	got := collectFixture(t, events...)
	if content := got["content"].([]any); len(content) != 1 || content[0].(map[string]any)["text"] != "hello" {
		t.Fatalf("content = %#v", content)
	}
	usage := got["usage"].(map[string]any)
	if usage["input_tokens"] != 1 || usage["cache_creation_input_tokens"] != 6 ||
		usage["cache_read_input_tokens"] != 0 || usage["output_tokens"] != 2 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestAnthropicCollectorAdapterToolPreservesLargeInteger(t *testing.T) {
	const largeInteger int64 = 9007199254740993
	var adapter anthropicAdapter
	events, err := adapter.Start("m1", "test", 0)
	if err != nil {
		t.Fatal(err)
	}
	feed, err := adapter.Feed([]outputparser.Event{{Kind: "tool_call", Call: &outputparser.ToolCall{
		ID: "t1", Name: "f", Arguments: map[string]any{"value": largeInteger},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, feed...)
	finish, err := adapter.Finish("stop", nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, finish...)
	got := collectFixture(t, events...)
	block := got["content"].([]any)[0].(map[string]any)
	input := block["input"].(map[string]any)
	if input["value"] != json.Number("9007199254740993") {
		t.Fatalf("input value = %#v, want exact JSON integer", input["value"])
	}
	usage := got["usage"].(map[string]any)
	if usage["input_tokens"] != 0 || usage["output_tokens"] != 1 {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestAnthropicCollectorEmptyContentIsArray(t *testing.T) {
	got := collectFixture(t, started())
	content, ok := got["content"].([]any)
	if !ok || content == nil || len(content) != 0 {
		t.Fatalf("content = %#v, want empty array", got["content"])
	}
}
