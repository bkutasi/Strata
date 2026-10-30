package responses

import (
	"errors"
	"testing"
	"time"
)

func finishTestAssembler(t *testing.T, format, validatorError string) *Assembler {
	t.Helper()
	raw := []byte(`{"input":"hello"}`)
	if format != "" {
		raw = []byte(`{"input":"hello","text":{"format":{"type":"json_object"}}}`)
	}
	req, err := Decode(raw)
	if err != nil {
		t.Fatal(err)
	}
	validate := func(text, finish string) (string, error) {
		if validatorError != "" {
			return "", errors.New(validatorError)
		}
		return text, nil
	}
	a, err := NewAssembler(req, Prepared{}, AssemblerOptions{
		Model: "test", PromptTokens: 12, Validate: validate,
		ID:  func(prefix string) (string, error) { return prefix + "_test", nil },
		Now: func() time.Time { return time.Unix(100, 0) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Start(); err != nil {
		t.Fatal(err)
	}
	return a
}

func TestAssemblerFinishNaturalAndLength(t *testing.T) {
	for _, tc := range []struct {
		finish, status, event string
	}{
		{"stop", "completed", "response.completed"},
		{"length", "incomplete", "response.incomplete"},
	} {
		t.Run(tc.finish, func(t *testing.T) {
			a := finishTestAssembler(t, "", "")
			a.item = map[string]any{"id": "msg_test", "type": "message", "status": "in_progress",
				"role": "assistant", "content": []any{map[string]any{"type": "output_text",
					"text": "answer", "annotations": []any{}, "logprobs": []any{}}}}
			a.response["output"] = []any{a.item}
			prompt, completion, reasoning, reused := int64(8), int64(5), int64(2), int64(20)
			events, err := a.Finish(Done{Finish: tc.finish, CompletionTokens: completion, PromptTokens: &prompt,
				ReasoningTokens: reasoning, Reused: reused, Timings: map[string]any{"total": 1}})
			if err != nil {
				t.Fatal(err)
			}
			final := events[len(events)-1]
			if final["type"] != tc.event {
				t.Fatalf("terminal event = %v, want %s", final["type"], tc.event)
			}
			resp := final["response"].(map[string]any)
			if resp["status"] != tc.status {
				t.Fatalf("status = %v, want %s", resp["status"], tc.status)
			}
			usage := resp["usage"].(map[string]any)
			if usage["input_tokens"] != prompt || usage["output_tokens"] != completion || usage["total_tokens"] != int64(13) ||
				usage["input_tokens_details"].(map[string]any)["cached_tokens"] != prompt ||
				usage["output_tokens_details"].(map[string]any)["reasoning_tokens"] != reasoning {
				t.Fatalf("unexpected usage: %#v", usage)
			}
			if final["timings"].(map[string]any)["total"] != 1 {
				t.Fatalf("timings missing: %#v", final)
			}
			if tc.finish == "length" && resp["incomplete_details"].(map[string]any)["reason"] != "max_output_tokens" {
				t.Fatalf("incomplete details: %#v", resp["incomplete_details"])
			}
			if _, err := a.Finish(Done{}); err == nil {
				t.Fatal("second Finish succeeded")
			}
		})
	}
}

func TestAssemblerFinishIncompleteToolAndFailed(t *testing.T) {
	a := finishTestAssembler(t, "", "")
	a.item = map[string]any{"id": "fc_test", "type": "function_call", "status": "in_progress",
		"call_id": "call", "name": "tool", "arguments": `{"x":`}
	a.response["output"] = []any{a.item}
	events, err := a.Finish(Done{Finish: "stop"})
	if err != nil {
		t.Fatal(err)
	}
	if a.response["output"].([]any)[0].(map[string]any)["status"] != "incomplete" {
		t.Fatal("open tool was not marked incomplete")
	}
	if events[len(events)-1]["type"] != "response.completed" {
		t.Fatalf("terminal event = %v", events[len(events)-1]["type"])
	}

	b := finishTestAssembler(t, "", "")
	b.item = map[string]any{"id": "msg_test", "type": "message", "status": "in_progress",
		"content": []any{map[string]any{"type": "output_text", "text": "partial", "annotations": []any{}, "logprobs": []any{}}}}
	b.response["output"] = []any{b.item}
	event, err := b.Failed("broken", "")
	if err != nil {
		t.Fatal(err)
	}
	if event["type"] != "response.failed" || b.response["status"] != "failed" ||
		b.response["error"].(map[string]any)["code"] != "server_error" ||
		b.response["output"].([]any)[0].(map[string]any)["status"] != "incomplete" {
		t.Fatalf("unexpected failed response: %#v", event)
	}
}

func TestAssemblerFinishReturnsValidatorError(t *testing.T) {
	a := finishTestAssembler(t, "json", "invalid structured output")
	a.jsonText.WriteString(`{"x":1}`)
	_, err := a.Finish(Done{Finish: "stop"})
	if err == nil || err.Error() != "invalid structured output" {
		t.Fatalf("Finish error = %v", err)
	}
	if _, err := a.Finish(Done{}); err == nil {
		t.Fatal("Finish after validator error succeeded")
	}
}
