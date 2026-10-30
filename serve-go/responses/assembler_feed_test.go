package responses

import (
	"reflect"
	"testing"
	"time"

	"strata/requestnorm"
)

func newFeedTestAssembler(t *testing.T) *Assembler {
	t.Helper()
	req, err := Decode([]byte(`{"input":"hi","tools":[{"type":"function","name":"run"},{"type":"custom","name":"shell"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := Prepare(req, PrepareOptions{Family: requestnorm.FamilyQwen})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	a, err := NewAssembler(req, prepared, AssemblerOptions{
		Model: "test-model", Now: func() time.Time { return time.Unix(1, 0) },
		ID: func(prefix string) (string, error) {
			n++
			return prefix + "_" + string(rune('a'+n-1)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Start(); err != nil {
		t.Fatal(err)
	}
	return a
}

func feed(t *testing.T, a *Assembler, ev ParserEvent) []Event {
	t.Helper()
	out, err := a.Feed(ev)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func eventTypes(events []Event) []string {
	out := make([]string, len(events))
	for i, ev := range events {
		out[i] = ev["type"].(string)
	}
	return out
}

func TestAssemblerFeedTextAndReasoningDeltas(t *testing.T) {
	a := newFeedTestAssembler(t)
	var got []string
	for _, ev := range []ParserEvent{
		{Kind: "reasoning", Text: "think "},
		{Kind: "reasoning", Text: "more"},
		{Kind: "content", Text: "hello"},
		{Kind: "content", Text: " world"},
	} {
		got = append(got, eventTypes(feed(t, a, ev))...)
	}
	want := []string{
		"response.output_item.added", "response.reasoning_text.delta",
		"response.reasoning_text.delta", "response.reasoning_text.done", "response.output_item.done",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.delta",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("event types = %v, want %v", got, want)
	}
	output := a.response["output"].([]any)
	reasoning := output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	text := output[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	if reasoning != "think more" || text != "hello world" {
		t.Fatalf("assembled text = (%v, %v)", reasoning, text)
	}
}

func TestAssemblerFeedFunctionAndCustomCalls(t *testing.T) {
	a := newFeedTestAssembler(t)
	callArgs := parse(`{"cmd":"pwd"}`)
	function := &Call{ID: "call_1", Name: "run", Arguments: callArgs}
	got := eventTypes(feed(t, a, ParserEvent{Kind: "tool_start", Call: function}))
	got = append(got, eventTypes(feed(t, a, ParserEvent{Kind: "tool_args", Call: function, Text: `{"cmd":`}))...)
	got = append(got, eventTypes(feed(t, a, ParserEvent{Kind: "tool_args", Call: function, Text: `"pwd"}`}))...)
	got = append(got, eventTypes(feed(t, a, ParserEvent{Kind: "tool_call", Call: function}))...)

	custom := &Call{ID: "call_2", Name: "shell", Arguments: parse(`{"input":"echo hi"}`)}
	got = append(got, eventTypes(feed(t, a, ParserEvent{Kind: "tool_call", Call: custom}))...)
	want := []string{
		"response.output_item.added",
		"response.function_call_arguments.delta", "response.function_call_arguments.delta",
		"response.function_call_arguments.done", "response.output_item.done",
		"response.output_item.added", "response.custom_tool_call_input.delta",
		"response.custom_tool_call_input.done", "response.output_item.done",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("event types = %v, want %v", got, want)
	}
	output := a.response["output"].([]any)
	fn := output[0].(map[string]any)
	ct := output[1].(map[string]any)
	if fn["arguments"] != `{"cmd":"pwd"}` || fn["status"] != "completed" {
		t.Fatalf("function call item = %#v", fn)
	}
	if ct["input"] != "echo hi" || ct["status"] != "completed" {
		t.Fatalf("custom tool call item = %#v", ct)
	}
}

func TestAssemblerFeedCustomCallInputFallbackMatchesPython(t *testing.T) {
	tests := []struct {
		name      string
		arguments string
		want      string
	}{
		{name: "missing input", arguments: `{"cmd":"café"}`, want: `{"cmd": "café"}`},
		{name: "non-string input", arguments: `{"input":7}`, want: `{"input": 7}`},
		{name: "non-object arguments", arguments: `[1,"café"]`, want: `[1, "café"]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := newFeedTestAssembler(t)
			call := &Call{ID: "call_fallback", Name: "shell", Arguments: parse(tt.arguments)}
			events := feed(t, a, ParserEvent{Kind: "tool_call", Call: call})
			var delta string
			for _, event := range events {
				if event["type"] == "response.custom_tool_call_input.delta" {
					delta = event["delta"].(string)
				}
			}
			item := a.response["output"].([]any)[0].(map[string]any)
			if delta != tt.want || item["input"] != tt.want {
				t.Fatalf("custom input delta/item = (%q, %v), want %q", delta, item["input"], tt.want)
			}
		})
	}
}

func TestAssemblerFeedProvisionalCallTruncation(t *testing.T) {
	a := newFeedTestAssembler(t)
	call := &Call{ID: "call_truncated", Name: "run", Arguments: parse(`{}`)}
	feed(t, a, ParserEvent{Kind: "tool_start", Call: call})
	feed(t, a, ParserEvent{Kind: "tool_args", Call: call, Text: `{"cmd":`})
	got := feed(t, a, ParserEvent{Kind: "content", Text: "continuation"})
	want := []string{
		"response.function_call_arguments.done", "response.output_item.done",
		"response.output_item.added", "response.content_part.added", "response.output_text.delta",
	}
	if types := eventTypes(got); !reflect.DeepEqual(types, want) {
		t.Fatalf("event types = %v, want %v", types, want)
	}
	item := a.response["output"].([]any)[0].(map[string]any)
	if item["status"] != "incomplete" || item["arguments"] != `{"cmd":` {
		t.Fatalf("truncated call item = %#v", item)
	}
}
