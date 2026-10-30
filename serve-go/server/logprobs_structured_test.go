package server

import (
	"reflect"
	"testing"
)

func TestStructuredLogprobs(t *testing.T) {
	entries := []logprobEntry{{Token: "```json\n"}, {Token: `{"x": `}, {Token: `1}`}, {Token: "\n```"}}
	raw := "```json\n{\"x\": 1}\n```"
	content, got := structuredLogprobs(raw, `{"x":1}`, entries)
	if content != `{"x": 1}` || !reflect.DeepEqual(got, entries[1:3]) {
		t.Fatalf("content=%q entries=%#v", content, got)
	}
	content, got = structuredLogprobs(raw, `{"x":2}`, entries)
	if content != `{"x":2}` || got != nil {
		t.Fatalf("rewritten content=%q entries=%#v", content, got)
	}
	if got := logprobSpan(raw, `"x"`, entries); got != nil {
		t.Fatalf("accepted span inside a token: %#v", got)
	}
}
