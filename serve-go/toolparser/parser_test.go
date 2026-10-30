package toolparser

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func numberTree(v any) any {
	switch x := v.(type) {
	case string:
		codepoints := []int{}
		for _, r := range x {
			codepoints = append(codepoints, int(r))
		}
		return map[string]any{"string": codepoints}
	case json.Number:
		return map[string]any{"integer": string(x)}
	case float64:
		return map[string]any{"float_bits": fmt.Sprintf("%016x", math.Float64bits(x))}
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = numberTree(item)
		}
		return out
	case map[string]any:
		keys := make([]string, 0, len(x))
		for key := range x {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out := make([]any, 0, len(x))
		for _, key := range keys {
			out = append(out, []any{numberTree(key), numberTree(x[key])})
		}
		return map[string]any{"object": out}
	}
	return v
}

func TestNonFiniteJSONSafety(t *testing.T) {
	call, err := ParseToolCall(`<function=f><parameter=x>[NaN,{"n":-Infinity},1e400]</parameter></function>`, nil)
	if err != nil {
		t.Fatal(err)
	}
	values := call.Arguments["x"].([]any)
	if !math.IsNaN(values[0].(float64)) || !math.IsInf(values[1].(map[string]any)["n"].(float64), -1) || !math.IsInf(values[2].(float64), 1) {
		t.Fatalf("unexpected values: %#v", values)
	}
	if _, err := json.Marshal(call); err == nil {
		t.Fatal("non-finite numbers must not silently become HTTP JSON")
	}
}

func TestUnsupportedUnicode(t *testing.T) {
	for _, value := range []string{`"\ud800"`, `"\udc00"`, `{"\ud800":1}`, `{"x":"\ud800","x":1}`} {
		body := "<function=f><parameter=x>" + value + "</parameter></function>"
		if _, err := ParseToolCall(body, nil); !errors.Is(err, ErrUnsupportedUnicode) {
			t.Fatalf("%q: expected unsupported Unicode, got %v", value, err)
		}
	}
	if _, err := ParseToolCall(string([]byte{0xff}), nil); !errors.Is(err, ErrUnsupportedUnicode) {
		t.Fatalf("invalid UTF-8: %v", err)
	}
}

func TestFramingAndBody(t *testing.T) {
	body := "é<function=f><parameter=x>1</parameter></function></tool_call>"
	if got := CallEnd(body); got != 50 {
		t.Fatalf("CallEnd = %d", got)
	}
	if got := ParamEndOffset("x</parameter>", false); got != -2 {
		t.Fatalf("ParamEnd = %d", got)
	}
	if got := ParamEndOffset("x</parameter>", true); got != 1 {
		t.Fatalf("ParamEnd final = %d", got)
	}
	call, err := ParseToolCall("<function=f><parameter=x>\n123\n</parameter></function>",
		map[string]any{"parameters": map[string]any{"properties": map[string]any{"x": map[string]any{"type": "string"}}}})
	if err != nil || call.Name != "f" || call.Arguments["x"] != "123" {
		t.Fatalf("call = %#v, %v", call, err)
	}
}

func TestDifferential(t *testing.T) {
	python := filepath.Join("..", "..", ".venv", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Skip("project .venv Python unavailable")
	}
	type request struct {
		Op     string         `json:"op"`
		Text   string         `json:"text"`
		Final  bool           `json:"final"`
		Schema map[string]any `json:"schema,omitempty"`
		Reject bool           `json:"-"`
	}
	var requests []request
	texts := []string{
		"", "é\u2003<function=f><parameter=x>hello </parameter> quoted </tool_call> </function> more</parameter>\n</function> \n</tool_call>",
		"<function=f><parameter=x>{\"x\":\"</parameter>\"}</parameter><parameter=y>2</parameter></function></tool_call>",
		"not a call </tool_call>", "<function=x>bad</tool_call>",
	}
	for _, text := range texts {
		runes := []rune(text)
		for i := range len(runes) + 1 {
			part := string(runes[:i])
			requests = append(requests, request{Op: "param", Text: part}, request{Op: "param", Text: part, Final: true},
				request{Op: "function", Text: part}, request{Op: "call", Text: part})
		}
	}
	schema := map[string]any{"parameters": map[string]any{"properties": map[string]any{"x": map[string]any{"type": "string"}}}}
	for _, text := range []string{
		"<function=f><parameter=x>\n123\n</parameter><parameter=y>[1,true,null]</parameter></function>",
		"<function= f ><parameter=x>{\"a\": 1}</parameter><parameter=x>last</parameter></function>",
		"<function=f><parameter=x>abc</parameter> inside</parameter></function>",
		"<function=f><parameter=x>unfinished", "<function=f><parameter=broken", "bad",
	} {
		requests = append(requests, request{Op: "parse", Text: text}, request{Op: "parse", Text: text, Schema: schema})
	}
	for _, value := range []string{
		"NaN", "Infinity", "-Infinity", "1e400", "-1e400", "1e-4000", "-0", "-0.0",
		"1.234567890123456789", "9007199254740993", "1e308", "1e309",
		`[NaN,Infinity,-Infinity,1e400,{"v":[-1e400,2.5,0]}]`,
		`{"NaN":"Infinity","x":"__strata_nonfinite__NaN","y":NaN}`,
		`["\u005f_strata_nonfinite__NaN",NaN]`,
		`{"x":NaN,"x":3}`, "{NaN:1}", "{Infinity:1}", "{-Infinity:1}", `[{"x":{NaN:1}}]`,
		`"\ud83d\ude00"`, `{"\ud83d\ude00":"😀"}`, `"\\ud800"`,
		`"\ud800`, `{"\ud800":1,}`, "NaNx", "[Infinity,]", "01", "1e", "nan",
		strings.Repeat("9", 4300), strings.Repeat("9", 4301),
		`{"x":` + strings.Repeat("9", 4301) + `,"x":1}`,
	} {
		body := "<function=f><parameter=x>" + value + "</parameter></function>"
		requests = append(requests, request{Op: "parse_numbers", Text: body}, request{Op: "parse_numbers", Text: body, Schema: schema})
	}
	for _, value := range []string{`"\ud800"`, `{"\ud800":"\udc00"}`, `["\ud800"]`} {
		body := "<function=f><parameter=x>" + value + "</parameter></function>"
		requests = append(requests, request{Op: "parse", Text: body, Reject: true},
			request{Op: "parse", Text: body, Schema: schema})
	}
	var input bytes.Buffer
	enc := json.NewEncoder(&input)
	for _, r := range requests {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	cmd := exec.Command(python, "oracle.py")
	cmd.Stdin = &input
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("oracle: %v: %s", err, stderr.String())
	}
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for i, r := range requests {
		if !scanner.Scan() {
			t.Fatalf("oracle stopped at %d: %v", i, scanner.Err())
		}
		var response struct {
			Result json.RawMessage `json:"result"`
			Error  string          `json:"error"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		var actual any
		var parseErr error
		switch r.Op {
		case "param":
			actual = ParamEndOffset(r.Text, r.Final)
		case "function":
			actual = FunctionEnd(r.Text)
		case "call":
			actual = CallEnd(r.Text)
		case "parse":
			var call ToolCall
			call, parseErr = ParseToolCall(r.Text, r.Schema)
			actual = numberTree(map[string]any{"name": call.Name, "arguments": call.Arguments})
		case "parse_numbers":
			var call ToolCall
			call, parseErr = ParseToolCall(r.Text, r.Schema)
			actual = numberTree(map[string]any{"name": call.Name, "arguments": call.Arguments})
		}
		if r.Reject {
			if response.Error != "" || !strings.Contains(string(response.Result), "55296") || !errors.Is(parseErr, ErrUnsupportedUnicode) {
				t.Fatalf("surrogate reference %s; Go error %v", response.Result, parseErr)
			}
			continue
		}
		if (response.Error != "") != (parseErr != nil) {
			t.Fatalf("request %#v: error %v oracle %s", r, parseErr, response.Error)
		}
		if parseErr != nil {
			continue
		}
		raw, err := json.Marshal(actual)
		if err != nil {
			t.Fatal(err)
		}
		var got, want any
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(response.Result, &want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("request %#v: got %s want %s", r, raw, response.Result)
		}
	}
	if scanner.Scan() {
		t.Fatal("extra oracle response")
	}
}
