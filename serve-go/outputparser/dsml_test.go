package outputparser

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"
)

type dsmlCase struct {
	Parts []string `json:"parts"`
}

type dsmlParameter struct {
	Tool     string `json:"tool"`
	Name     string `json:"name"`
	IsString bool   `json:"is_string"`
}

type dsmlResult struct {
	Pieces []string        `json:"pieces"`
	Params []dsmlParameter `json:"params"`
}

func translateDSML(c dsmlCase) (dsmlResult, error) {
	result := dsmlResult{Pieces: []string{}, Params: []dsmlParameter{}}
	tr := NewDSMLTranslator(func(tool, name string, isString bool) {
		result.Params = append(result.Params, dsmlParameter{tool, name, isString})
	})
	for _, part := range c.Parts {
		piece, err := tr.Feed(part)
		if err != nil {
			return result, err
		}
		result.Pieces = append(result.Pieces, piece)
	}
	result.Pieces = append(result.Pieces, tr.Flush())
	return result, nil
}

func TestDSMLTranslation(t *testing.T) {
	text := `<｜DSML｜ calls><｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="id" string="true">A-17</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`
	got, err := translateDSML(dsmlCase{Parts: []string{text}})
	want := "<tool_call>\n<function=lookup>\n<parameter=id>\nA-17\n</parameter>\n</function>\n</tool_call>\n"
	if err != nil || strings.Join(got.Pieces, "") != want ||
		!reflect.DeepEqual(got.Params, []dsmlParameter{{"lookup", "id", true}}) {
		t.Fatalf("translation %#v, error %v", got, err)
	}
	tr := NewDSMLTranslator(nil)
	if _, err := tr.Feed(string([]byte{0xff})); !errors.Is(err, ErrInvalidUTF8) {
		t.Fatalf("invalid UTF-8: %v", err)
	}
}

func TestDSMLTranslatorMatchesPython(t *testing.T) {
	samples := []string{
		`<｜DSML｜ calls><｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="id" string="true">A-17</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`,
		" \n<｜DSML｜ function_calls> \n<｜DSML｜ invoke name=\"f\"><｜DSML｜ parameter name=\"x\" string=\"false\">3</｜DSML｜ parameter></｜DSML｜ invoke> \n</｜DSML｜ function_calls>",
		`<｜DSML｜tool_calls><｜DSML｜invoke name="f"><｜DSML｜parameter name="x">é🙂</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`,
		`<｜DSML｜ calls><｜DSML｜ invoke name="a"></｜DSML｜ invoke><｜DSML｜ invoke name="b"></｜DSML｜ invoke></｜DSML｜ calls>`,
		`x <div> y </｜DSML｜ nope> <｜User｜> <｜DSML｜ unknown>`,
		`<｜DSML｜ invoke name="partial`,
		`<｜DSML｜ parameter name="x" string="bad">v</｜DSML｜ parameter>`,
		`<｜DSML｜ calls>` + strings.Repeat(" ", 4) + `<｜DSML｜ invoke name="f">` + strings.Repeat(" ", 4),
		`<｜DSML｜ ` + strings.Repeat("x", 601),
		`</｜DSML｜ parameter><｜DSML｜ parameter name="orphan">v`,
	}
	var cases []dsmlCase
	for _, text := range samples {
		runes := []rune(text)
		for i := range len(runes) + 1 {
			cases = append(cases, dsmlCase{Parts: []string{string(runes[:i]), string(runes[i:])}})
		}
		parts := make([]string, len(runes))
		for i, r := range runes {
			parts[i] = string(r)
		}
		cases = append(cases, dsmlCase{Parts: parts})
	}
	const oracle = `
import json, os, sys
sys.path.insert(0, os.path.abspath("../.."))
from serve.dsml import DsmlTranslator
results = []
for case in json.load(sys.stdin):
    params = []
    tr = DsmlTranslator(lambda tool, name, is_string: params.append(
        {"tool": tool or "", "name": name, "is_string": is_string}))
    pieces = [tr.feed(part) for part in case["parts"]]
    pieces.append(tr.flush())
    results.append({"pieces": pieces, "params": params})
json.dump(results, sys.stdout, ensure_ascii=False)
`
	data, err := json.Marshal(cases)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python(t), "-c", oracle)
	cmd.Stdin = bytes.NewReader(data)
	gotPython, err := cmd.Output()
	if err != nil {
		t.Fatalf("Python oracle: %v", err)
	}
	var expected []dsmlResult
	if err := json.Unmarshal(gotPython, &expected); err != nil || len(expected) != len(cases) {
		t.Fatalf("Python results: %d, error %v", len(expected), err)
	}
	for i, c := range cases {
		got, err := translateDSML(c)
		if err != nil || !reflect.DeepEqual(got, expected[i]) {
			t.Fatalf("case %d %s: got %#v, want %#v, error %v",
				i, fmt.Sprint(c.Parts), got, expected[i], err)
		}
	}
}

func TestDeepSeekParserMatchesPython(t *testing.T) {
	const call = `<｜DSML｜ calls><｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="id" string="true">123</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`
	tools := []map[string]any{{"name": "lookup", "parameters": map[string]any{"properties": map[string]any{
		"id": map[string]any{"type": "integer"},
	}}}}
	var cases []request
	for _, c := range []struct {
		text     string
		thinking bool
		finish   string
	}{
		{call, false, "stop"},
		{strings.Replace(call, `string="true"`, `string="false"`, 1), false, "stop"},
		{"reason</think>" + call, true, "stop"},
		{"answer without closing tag", true, "stop"},
		{"truncated reasoning", true, "length"},
		{"ordinary</think>answer", true, "stop"},
	} {
		for _, stream := range []bool{false, true} {
			addSplits(&cases, request{Thinking: c.thinking, Tools: tools, Stream: stream, Finish: c.finish}, c.text, true)
		}
	}
	const oracle = `
import json, os, sys
sys.path.insert(0, os.path.abspath("../.."))
from serve.dsml import DsmlOutputParser
def tree(value):
    if value is None or isinstance(value, bool): return value
    if isinstance(value, str): return {"string": [ord(c) for c in value]}
    if isinstance(value, int): return {"integer": str(value)}
    if isinstance(value, list): return [tree(item) for item in value]
    return {"object": [[tree(key), tree(value[key])] for key in sorted(value)]}
results = []
for c in json.load(sys.stdin):
    p = DsmlOutputParser(thinking=c["thinking"], tools=c["tools"],
                         stream_tools=c["stream"], declared_only=c["declared_only"])
    ids = {}
    def norm(events):
        out = []
        for event in events:
            call = None
            if event.call is not None:
                ident = ids.setdefault(event.call.id, len(ids) + 1)
                call = {"name": event.call.name, "arguments": tree(event.call.arguments), "id": ident}
            out.append({"kind": event.kind, "text": event.text, "call": call})
        return out
    batches = [norm(p.feed(part)) for part in c["parts"]]
    batches.append(norm(p.finish(c["finish"] or None)))
    results.append(batches)
json.dump(results, sys.stdout)
`
	data, err := json.Marshal(cases)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python(t), "-c", oracle)
	cmd.Stdin = bytes.NewReader(data)
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("Python oracle: %v", err)
	}
	var want []any
	if err := json.Unmarshal(output, &want); err != nil || len(want) != len(cases) {
		t.Fatalf("Python results: %d, error %v", len(want), err)
	}
	for i, c := range cases {
		n := 0
		parser, err := NewDeepSeek(Options{Thinking: c.Thinking, Tools: c.Tools, StreamTools: c.Stream,
			DeclaredOnly: c.DeclaredOnly, ID: func() (string, error) {
				n++
				return fmt.Sprintf("call_%d", n), nil
			}})
		if err != nil {
			t.Fatal(err)
		}
		ids := map[string]int{}
		norm := func(events []Event) []any {
			out := []any{}
			for _, event := range events {
				var call any
				if event.Call != nil {
					id, ok := ids[event.Call.ID]
					if !ok {
						id = len(ids) + 1
						ids[event.Call.ID] = id
					}
					call = map[string]any{"name": event.Call.Name, "arguments": tree(event.Call.Arguments), "id": id}
				}
				out = append(out, map[string]any{"kind": event.Kind, "text": event.Text, "call": call})
			}
			return out
		}
		batches := []any{}
		for _, part := range c.Parts {
			events, err := parser.Feed(part)
			if err != nil {
				t.Fatalf("case %d: %v", i, err)
			}
			batches = append(batches, norm(events))
		}
		events, err := parser.Finish(c.Finish)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		batches = append(batches, norm(events))
		encoded, err := json.Marshal(batches)
		if err != nil {
			t.Fatal(err)
		}
		var got any
		if err := json.Unmarshal(encoded, &got); err != nil || !reflect.DeepEqual(got, want[i]) {
			t.Fatalf("case %d parts %q: got %s, want %v, error %v", i, c.Parts, encoded, want[i], err)
		}
	}
}

func TestDeepSeekParserErrorIsTerminal(t *testing.T) {
	parser, err := NewDeepSeek(Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parser.Feed(string([]byte{0xff})); !errors.Is(err, ErrInvalidUTF8) {
		t.Fatalf("invalid UTF-8: %v", err)
	}
	if _, err := parser.Feed("valid"); !errors.Is(err, ErrClosed) {
		t.Fatalf("feed after error: %v", err)
	}
	if _, err := parser.Finish("stop"); !errors.Is(err, ErrClosed) {
		t.Fatalf("finish after error: %v", err)
	}
}
