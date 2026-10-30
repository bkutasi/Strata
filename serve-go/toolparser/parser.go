// Package toolparser implements the tool-call body and framing primitives in
// serve/frontend.py. It does not implement frontend.OutputParser.
package toolparser

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	ParamStart = "<parameter="
	ParamEnd   = "</parameter>"
	FuncStart  = "<function="
	FuncEnd    = "</function>"
	CallEndTag = "</tool_call>"
)

// Python's str.isspace includes four ASCII information separators that Go's
// unicode.IsSpace does not. Python indexes strings by Unicode code point.
func space(r rune) bool         { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }
func lstrip(s string) string    { return strings.TrimLeftFunc(s, space) }
func strip(s string) string     { return strings.TrimFunc(s, space) }
func chars(s string, n int) int { return utf8.RuneCountInString(s[:n]) }

// paramEndByte is the byte-offset counterpart used internally.
func paramEndByte(text string, final bool) int {
	for pos := 0; pos <= len(text); {
		rel := strings.Index(text[pos:], ParamEnd)
		if rel < 0 {
			return -1
		}
		at := pos + rel
		after := lstrip(text[at+len(ParamEnd):])
		if strings.HasPrefix(after, ParamStart) || strings.HasPrefix(after, FuncEnd) {
			return at
		}
		if after == "" || strings.HasPrefix(ParamStart, after) || strings.HasPrefix(FuncEnd, after) {
			if final {
				return at
			}
			return -2
		}
		pos = at + 1
	}
	return -1
}

// ParamEnd returns a Python character offset, -1 if absent, or -2 if pending.
func ParamEndOffset(text string, final bool) int {
	n := paramEndByte(text, final)
	if n < 0 {
		return n
	}
	return chars(text, n)
}

// CallEnd returns the Python character offset of the closing tool-call tag.
func CallEnd(text string) int {
	n := callEndByte(text)
	if n < 0 {
		return n
	}
	return chars(text, n)
}

func callEndByte(text string) int {
	s := lstrip(text)
	pos := len(text) - len(s)
	if !strings.HasPrefix(s, FuncStart) {
		if strings.HasPrefix(FuncStart, s) {
			return -1
		}
		return strings.Index(text, CallEndTag)
	}
	gt := strings.IndexByte(text[pos:], '>')
	if gt < 0 {
		return -1
	}
	pos += gt + 1
	for {
		rest := text[pos:]
		s = lstrip(rest)
		pos += len(rest) - len(s)
		if strings.HasPrefix(s, ParamStart) {
			gt = strings.IndexByte(text[pos:], '>')
			if gt < 0 {
				return -1
			}
			gt += pos
			end := paramEndByte(text[gt+1:], false)
			if end < 0 {
				return -1
			}
			pos = gt + 1 + end + len(ParamEnd)
		} else if strings.HasPrefix(s, FuncEnd) {
			rest = text[pos+len(FuncEnd):]
			s = lstrip(rest)
			if strings.HasPrefix(s, CallEndTag) {
				return pos + len(FuncEnd) + len(rest) - len(s)
			}
			if strings.HasPrefix(CallEndTag, s) {
				return -1
			}
			n := strings.Index(text[pos:], CallEndTag)
			if n < 0 {
				return -1
			}
			return pos + n
		} else if s == "" || strings.HasPrefix(ParamStart, s) || strings.HasPrefix(FuncEnd, s) {
			return -1
		} else {
			n := strings.Index(text[pos:], CallEndTag)
			if n < 0 {
				return -1
			}
			return pos + n
		}
	}
}

// FunctionEnd returns the Python character offset just after </function>.
func FunctionEnd(text string) int {
	s := lstrip(text)
	pos := len(text) - len(s)
	if !strings.HasPrefix(s, FuncStart) {
		return -1
	}
	gt := strings.IndexByte(text[pos:], '>')
	if gt < 0 {
		return -1
	}
	pos += gt + 1
	for {
		rest := text[pos:]
		s = lstrip(rest)
		pos += len(rest) - len(s)
		if strings.HasPrefix(s, ParamStart) {
			gt = strings.IndexByte(text[pos:], '>')
			if gt < 0 {
				return -1
			}
			gt += pos
			end := paramEndByte(text[gt+1:], false)
			if end < 0 {
				return -1
			}
			pos = gt + 1 + end + len(ParamEnd)
		} else if strings.HasPrefix(s, FuncEnd) {
			return chars(text, pos+len(FuncEnd))
		} else {
			return -1
		}
	}
}

type ToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ErrUnsupportedUnicode indicates a Python string that cannot be represented
// losslessly by the UTF-8 strings used by this API.
var ErrUnsupportedUnicode = errors.New("tool call contains unsupported Unicode")

// ParseToolCall parses the body, without generating Python's random call ID.
// Schema follows the Python tool schema, not the bare parameters schema.
func ParseToolCall(body string, schema map[string]any) (ToolCall, error) {
	if !utf8.ValidString(body) {
		return ToolCall{}, ErrUnsupportedUnicode
	}
	body = strip(body)
	gt := strings.IndexByte(body, '>')
	if !strings.HasPrefix(body, FuncStart) || gt < 0 {
		return ToolCall{}, fmt.Errorf("malformed tool call: %s", string([]rune(body)[:min(80, len([]rune(body)))]))
	}
	call := ToolCall{Name: strip(body[len(FuncStart):gt]), Arguments: map[string]any{}}
	params, _ := schema["parameters"].(map[string]any)
	props, _ := params["properties"].(map[string]any)
	rest := body[gt+1:]
	for {
		at := strings.Index(rest, ParamStart)
		if at < 0 {
			break
		}
		rest = rest[at+len(ParamStart):]
		gt = strings.IndexByte(rest, '>')
		if gt < 0 {
			return ToolCall{}, fmt.Errorf("substring not found")
		}
		name := rest[:gt]
		rest = rest[gt+1:]
		end := paramEndByte(rest, true)
		value := rest
		if end >= 0 {
			value, rest = rest[:end], rest[end+len(ParamEnd):]
		} else {
			rest = ""
		}
		value = strings.TrimPrefix(value, "\n")
		value = strings.TrimSuffix(value, "\n")
		prop, _ := props[name].(map[string]any)
		if declared, ok := prop["type"].(string); ok && declared == "string" {
			call.Arguments[name] = value
			continue
		}
		if decoded, ok := pythonJSON(value); ok {
			if hasUnpairedSurrogate(value) {
				return ToolCall{}, ErrUnsupportedUnicode
			}
			call.Arguments[name] = decoded
			continue
		}
		call.Arguments[name] = value
	}
	return call, nil
}

// Called only after JSON parsing succeeds. Escaped backslashes are not Unicode
// escapes; adjacent high/low escapes make one representable Unicode scalar.
func hasUnpairedSurrogate(text string) bool {
	inString := false
	for i := 0; i < len(text); i++ {
		if text[i] == '"' {
			inString = !inString
			continue
		}
		if !inString || text[i] != '\\' {
			continue
		}
		i++
		if i >= len(text) || text[i] != 'u' || i+4 >= len(text) {
			continue
		}
		n, err := strconv.ParseUint(text[i+1:i+5], 16, 16)
		if err != nil {
			continue
		}
		i += 4
		if n >= 0xdc00 && n <= 0xdfff {
			return true
		}
		if n >= 0xd800 && n <= 0xdbff {
			if i+6 >= len(text) || text[i+1:i+3] != `\u` {
				return true
			}
			low, err := strconv.ParseUint(text[i+3:i+7], 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return true
			}
			i += 6
		}
	}
	return false
}

// pythonJSON matches json.loads number semantics. Integers are json.Number
// (arbitrary precision); floats are float64, including non-finite values.
// encoding/json deliberately rejects non-finite floats, so HTTP consumers
// must choose an explicit policy rather than emit non-standard JSON.
func pythonJSON(text string) (any, bool) {
	marker := "__strata_nonfinite__"
	// Escaped JSON strings can decode to a marker not present in raw input.
	// Choose a marker absent from both the raw input and every decoded string.
	var literals []string
	for i := 0; i < len(text); i++ {
		if text[i] != '"' {
			continue
		}
		start := i
		for i++; i < len(text); i++ {
			if text[i] == '\\' {
				i++
				continue
			}
			if text[i] == '"' {
				break
			}
		}
		if i < len(text) {
			var s string
			if json.Unmarshal([]byte(text[start:i+1]), &s) == nil {
				literals = append(literals, s)
			}
		}
	}
	for {
		collision := strings.Contains(text, marker)
		for _, s := range literals {
			collision = collision || strings.Contains(s, marker)
		}
		if !collision {
			break
		}
		marker += "_"
	}
	var out strings.Builder
	inString, escaped := false, false
	for i := 0; i < len(text); {
		c := text[i]
		if inString {
			out.WriteByte(c)
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				inString = false
			}
			i++
			continue
		}
		if c == '"' {
			inString = true
			out.WriteByte(c)
			i++
			continue
		}
		replaced := false
		for _, token := range []string{"-Infinity", "Infinity", "NaN"} {
			if strings.HasPrefix(text[i:], token) {
				out.WriteString(strconv.Quote(marker + token))
				i += len(token)
				replaced = true
				break
			}
		}
		if !replaced {
			out.WriteByte(c)
			i++
		}
	}
	decoder := json.NewDecoder(strings.NewReader(out.String()))
	decoder.UseNumber()
	v, ok := pythonValue(decoder, marker)
	if !ok {
		return nil, false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, false
	}
	return v, true
}

func pythonValue(decoder *json.Decoder, marker string) (any, bool) {
	v, err := decoder.Token()
	if err != nil {
		return nil, false
	}
	switch x := v.(type) {
	case json.Delim:
		switch x {
		case '[':
			items := []any{}
			for decoder.More() {
				item, ok := pythonValue(decoder, marker)
				if !ok {
					return nil, false
				}
				items = append(items, item)
			}
			end, err := decoder.Token()
			return items, err == nil && end == json.Delim(']')
		case '{':
			items := map[string]any{}
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return nil, false
				}
				name, ok := key.(string)
				if !ok || strings.HasPrefix(name, marker) {
					return nil, false
				}
				item, ok := pythonValue(decoder, marker)
				if !ok {
					return nil, false
				}
				items[name] = item
			}
			end, err := decoder.Token()
			return items, err == nil && end == json.Delim('}')
		default:
			return nil, false
		}
	case string:
		switch x {
		case marker + "NaN":
			return math.Float64frombits(0x7ff8000000000000), true
		case marker + "Infinity":
			return math.Inf(1), true
		case marker + "-Infinity":
			return math.Inf(-1), true
		}
	case json.Number:
		s := string(x)
		if !strings.ContainsAny(s, ".eE") {
			// CPython's default integer-string conversion limit is 4300 digits.
			if len(strings.TrimPrefix(s, "-")) > 4300 {
				return nil, false
			}
			n, ok := new(big.Int).SetString(s, 10)
			if !ok {
				return nil, false
			}
			return json.Number(n.String()), true
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			if e, ok := err.(*strconv.NumError); !ok || e.Err != strconv.ErrRange {
				return nil, false
			}
		}
		return f, true
	}
	return v, true
}
