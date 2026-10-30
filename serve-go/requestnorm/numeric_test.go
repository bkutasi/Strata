package requestnorm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// numericShape makes object order, integer/float distinction and the exact
// binary64 bits observable without relying on either side's float formatting.
func numericShape(v Value) any {
	switch v.Kind() {
	case Null:
		return nil
	case Boolean:
		b, _ := v.Bool()
		return b
	case String:
		s, _ := v.Text()
		return s
	case Number:
		if integer, ok := v.Integer(); ok {
			return []any{"int", integer}
		}
		f, _ := v.Float64()
		return []any{"float", fmt.Sprintf("%016x", math.Float64bits(f))}
	case Array:
		items := make([]any, 0, len(v.Items()))
		for _, item := range v.Items() {
			items = append(items, numericShape(item))
		}
		return []any{"array", items}
	case Object:
		fields := make([]any, 0, len(v.Fields()))
		for _, field := range v.Fields() {
			fields = append(fields, []any{field.Name, numericShape(field.Value)})
		}
		return []any{"object", fields}
	default:
		panic("unexpected JSON kind")
	}
}

func TestNumericSemanticsDifferential(t *testing.T) {
	python := filepath.Join("..", "..", ".venv", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Skip("project .venv Python unavailable")
	}
	// The same values travel through an object argument, a JSON-encoded
	// argument, and a tool schema, exactly as the frontend receives them.
	payload := `{"rounded":9007199254740993.0,"exact_int":9007199254740993,` +
		`"big_int":1234567890123456789012345678901234567890,` +
		`"nested":[1e-999,-1e-999,-0,-0.0,1e0,0.1,` +
		`1.00000000000000011102230246251565404236316680908203125,` +
		`5e-324,2.2250738585072014e-308,1.7976931348623157e308]}`
	request := `{"messages":[{"role":"assistant","tool_calls":[` +
		`{"function":{"name":"object_arg","arguments":` + payload + `}},` +
		`{"function":{"name":"string_arg","arguments":` + encodeJSON(t, payload) + `}}]}],` +
		`"tools":[{"name":"numbers","parameters":` + payload + `}]}`

	const script = `
import json, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from serve.frontend import openai_to_messages

def shape(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", struct.pack(">d", value).hex()]
    if isinstance(value, list):
        return ["array", [shape(item) for item in value]]
    return ["object", [[key, shape(item)] for key, item in value.items()]]

messages, tools, kwargs = openai_to_messages(json.load(sys.stdin))
json.dump(shape({"messages": messages, "tools": tools, "kwargs": kwargs}), sys.stdout)
`
	cmd := exec.Command(python, "-c", script)
	cmd.Stdin = strings.NewReader(request)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("Python reference: %v: %s", err, stderr.String())
	}
	var want any
	if err := json.Unmarshal(output, &want); err != nil {
		t.Fatalf("Python reference result %q: %v", output, err)
	}

	req, err := DecodeOpenAI([]byte(request))
	if err != nil {
		t.Fatal(err)
	}
	n, _ := NewNormalizer(FamilyQwen)
	result, err := n.NormalizeOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	// Compare the actual values handed to the renderer, before any marshaling.
	calls := result.Messages[0].ToolCalls
	actual := objectValue(
		Field{Name: "messages", Value: Value{kind: Array, items: []Value{
			objectValue(
				Field{Name: "role", Value: stringValue(result.Messages[0].Role)},
				Field{Name: "content", Value: stringValue(result.Messages[0].Content.Text)},
				Field{Name: "tool_calls", Value: Value{kind: Array, items: []Value{
					objectValue(Field{Name: "function", Value: objectValue(
						Field{Name: "name", Value: stringValue(calls[0].Function.Name)},
						Field{Name: "arguments", Value: calls[0].Function.Arguments},
					)}),
					objectValue(Field{Name: "function", Value: objectValue(
						Field{Name: "name", Value: stringValue(calls[1].Function.Name)},
						Field{Name: "arguments", Value: calls[1].Function.Arguments},
					)}),
				}}},
			),
		}}},
		Field{Name: "tools", Value: Value{kind: Array, items: result.Tools}},
		Field{Name: "kwargs", Value: result.Kwargs},
	)
	if got := numericShape(actual); !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.Marshal(got)
		t.Fatalf("numeric semantics:\nGo:        %s\nreference: %s", gotJSON, output)
	}

	// The public JSON form must round-trip to the same kinds and bits even
	// though it deliberately does not match Python's formatting byte-for-byte.
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := ParseValue(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got := numericShape(roundTrip); !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.Marshal(got)
		t.Fatalf("marshal changed numeric semantics:\nGo:        %s\nreference: %s", gotJSON, output)
	}
}

func TestNumericAccessorsAndOverflow(t *testing.T) {
	for _, tc := range []struct {
		input string
		want  any
	}{
		{`-0`, []any{"int", "0"}},
		{`9007199254740993`, []any{"int", "9007199254740993"}},
		{`9007199254740993.0`, []any{"float", "4340000000000000"}},
		{`1e-999`, []any{"float", "0000000000000000"}},
		{`-1e-999`, []any{"float", "8000000000000000"}},
		{`-0.0`, []any{"float", "8000000000000000"}},
		{`1.0`, []any{"float", "3ff0000000000000"}},
	} {
		v, err := ParseValue([]byte(tc.input))
		if err != nil {
			t.Fatalf("%s: %v", tc.input, err)
		}
		if got := numericShape(v); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v, want %v", tc.input, got, tc.want)
		}
		if text, ok := v.Text(); !ok || text != tc.input {
			t.Errorf("%s: original spelling = %q, %v", tc.input, text, ok)
		}
		_, isInteger := v.Integer()
		_, isFloat := v.Float64()
		if isInteger == isFloat {
			t.Errorf("%s: integer=%v, float=%v", tc.input, isInteger, isFloat)
		}
	}
	for _, input := range []string{`1e999`, `-1e999`} {
		if _, err := ParseValue([]byte(input)); err == nil {
			t.Errorf("%s: expected float overflow to be rejected", input)
		}
	}
}
