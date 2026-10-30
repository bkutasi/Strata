package requestnorm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ErrUnsupported identifies input which cannot be represented without loss.
var ErrUnsupported = errors.New("requestnorm: unsupported input")

// Kind identifies the JSON type of a Value.
type Kind uint8

const (
	Null Kind = iota
	Boolean
	Number
	String
	Array
	Object
)

// Field is an object member. Objects retain their input order, including
// nested JSON schemas; keys are unique.
type Field struct {
	Name  string
	Value Value
}

// Value is an ordered JSON value. Its zero value represents JSON null.
// Constructors copy their inputs and accessors return copies.
type Value struct {
	kind    Kind
	text    string
	flag    bool
	isFloat bool
	number  float64
	items   []Value
	fields  []Field
}

func (v Value) Kind() Kind { return v.kind }

// Text returns a JSON string or a number's original spelling and whether it
// has that kind. Number spellings are not necessarily the parsed value: use
// Integer or Float64 to access the Python-compatible numeric value.
func (v Value) Text() (string, bool) {
	return v.text, v.kind == String || v.kind == Number
}

// Integer returns a number parsed as an arbitrary-precision integer in decimal
// notation. JSON -0 is normalized to 0, as in Python's json.loads.
func (v Value) Integer() (string, bool) {
	if v.kind != Number || v.isFloat {
		return "", false
	}
	if v.text == "-0" {
		return "0", true
	}
	return v.text, true
}

// Float64 returns a number parsed as an IEEE 754 binary64 float. The boolean is
// false for integers, even if an integer can be exactly represented as a float.
// Signed zero and underflow are preserved.
func (v Value) Float64() (float64, bool) {
	return v.number, v.kind == Number && v.isFloat
}

// Bool returns a JSON boolean and whether it has that kind.
func (v Value) Bool() (bool, bool) { return v.flag, v.kind == Boolean }

// Items returns a copy of an array's elements, or nil for other kinds.
func (v Value) Items() []Value { return append([]Value(nil), v.items...) }

// Fields returns a copy of an object's members, or nil for other kinds.
func (v Value) Fields() []Field { return append([]Field(nil), v.fields...) }

// Lookup retrieves an object member.
func (v Value) Lookup(name string) (Value, bool) {
	for _, field := range v.fields {
		if field.Name == name {
			return field.Value, true
		}
	}
	return Value{}, false
}

func stringValue(s string) Value { return Value{kind: String, text: s} }
func objectValue(fields ...Field) Value {
	return Value{kind: Object, fields: append([]Field(nil), fields...)}
}

// ParseValue parses one standard JSON value, retaining object order and number
// spellings while storing decimal/exponent numbers as binary64 floats. It
// refuses duplicate keys, invalid Unicode, excessive integer digits and float
// overflow rather than silently changing their meaning.
func ParseValue(data []byte) (Value, error) {
	if err := checkUnicode(data); err != nil {
		return Value{}, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	v, err := readValue(decoder)
	if err != nil {
		return Value{}, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		if err == nil {
			return Value{}, errors.New("extra JSON value")
		}
		return Value{}, err
	}
	return v, nil
}

func readValue(d *json.Decoder) (Value, error) {
	token, err := d.Token()
	if err != nil {
		return Value{}, err
	}
	switch t := token.(type) {
	case nil:
		return Value{}, nil
	case bool:
		return Value{kind: Boolean, flag: t}, nil
	case string:
		return stringValue(t), nil
	case json.Number:
		s := string(t)
		if strings.ContainsAny(s, ".eE") {
			n, err := strconv.ParseFloat(s, 64)
			if err != nil || math.IsInf(n, 0) {
				return Value{}, fmt.Errorf("%w: floating-point overflow %s", ErrUnsupported, s)
			}
			return Value{kind: Number, text: s, isFloat: true, number: n}, nil
		} else if len(strings.TrimPrefix(s, "-")) > 4300 {
			return Value{}, fmt.Errorf("%w: integer exceeds 4300 digits", ErrUnsupported)
		}
		return Value{kind: Number, text: s}, nil
	case json.Delim:
		switch t {
		case '[':
			v := Value{kind: Array}
			for d.More() {
				item, err := readValue(d)
				if err != nil {
					return Value{}, err
				}
				v.items = append(v.items, item)
			}
			_, err := d.Token()
			return v, err
		case '{':
			v := Value{kind: Object}
			seen := make(map[string]bool)
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return Value{}, err
				}
				name, ok := key.(string)
				if !ok {
					return Value{}, errors.New("JSON object key is not a string")
				}
				if seen[name] {
					return Value{}, fmt.Errorf("%w: duplicate object key %q", ErrUnsupported, name)
				}
				seen[name] = true
				item, err := readValue(d)
				if err != nil {
					return Value{}, err
				}
				v.fields = append(v.fields, Field{Name: name, Value: item})
			}
			_, err := d.Token()
			return v, err
		}
	}
	return Value{}, errors.New("invalid JSON value")
}

// MarshalJSON retains object order, numeric types and the parsed binary64
// values. It does not implement Python json.dumps whitespace or float formatting.
func (v Value) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	if err := v.appendJSON(&b, false); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// pythonSpacing is used only for the reference's non-object argument fallback.
func (v Value) appendJSON(b *bytes.Buffer, pythonSpacing bool) error {
	switch v.kind {
	case Null:
		b.WriteString("null")
	case Boolean:
		b.WriteString(strconv.FormatBool(v.flag))
	case Number:
		if v.isFloat {
			if pythonSpacing {
				return fmt.Errorf("%w: float in non-object tool arguments", ErrUnsupported)
			}
			s := strconv.FormatFloat(v.number, 'g', -1, 64)
			if !strings.ContainsAny(s, ".eE") {
				s += ".0" // preserve float type for an integer-valued float
			}
			b.WriteString(s)
		} else {
			integer, _ := v.Integer()
			b.WriteString(integer)
		}
	case String:
		var encoded bytes.Buffer
		e := json.NewEncoder(&encoded)
		e.SetEscapeHTML(false)
		if err := e.Encode(v.text); err != nil {
			return err
		}
		s := strings.TrimSuffix(encoded.String(), "\n")
		// Python's ensure_ascii=False leaves these two code points unescaped.
		// Do not rewrite a literal "\\u2028" inside the input string.
		for i := 0; i < len(s); i++ {
			if s[i] == '\\' && i+1 < len(s) {
				if strings.HasPrefix(s[i:], `\u2028`) {
					b.WriteString("\u2028")
					i += 5
					continue
				}
				if strings.HasPrefix(s[i:], `\u2029`) {
					b.WriteString("\u2029")
					i += 5
					continue
				}
				b.WriteByte(s[i])
				i++
			}
			b.WriteByte(s[i])
		}
	case Array:
		b.WriteByte('[')
		for i, item := range v.items {
			if i > 0 {
				b.WriteByte(',')
				if pythonSpacing {
					b.WriteByte(' ')
				}
			}
			if err := item.appendJSON(b, pythonSpacing); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case Object:
		b.WriteByte('{')
		for i, field := range v.fields {
			if i > 0 {
				b.WriteByte(',')
				if pythonSpacing {
					b.WriteByte(' ')
				}
			}
			if err := stringValue(field.Name).appendJSON(b, pythonSpacing); err != nil {
				return err
			}
			b.WriteByte(':')
			if pythonSpacing {
				b.WriteByte(' ')
			}
			if err := field.Value.appendJSON(b, pythonSpacing); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("%w: invalid JSON kind", ErrUnsupported)
	}
	return nil
}

// encoding/json replaces lone surrogate escapes with U+FFFD; reject those
// escapes before decoding so a prompt is never silently changed.
func checkUnicode(data []byte) error {
	if !utf8.Valid(data) {
		return fmt.Errorf("%w: invalid UTF-8", ErrUnsupported)
	}
	inString := false
	for i := 0; i < len(data); i++ {
		switch data[i] {
		case '"':
			inString = !inString
		case '\\':
			if !inString || i+1 >= len(data) {
				continue
			}
			i++
			if data[i] != 'u' || i+4 >= len(data) {
				continue
			}
			n, err := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
			if err != nil {
				continue // the JSON decoder reports invalid escapes
			}
			i += 4
			if n >= 0xdc00 && n <= 0xdfff {
				return fmt.Errorf("%w: unpaired surrogate", ErrUnsupported)
			}
			if n >= 0xd800 && n <= 0xdbff {
				if i+6 >= len(data) || data[i+1] != '\\' || data[i+2] != 'u' {
					return fmt.Errorf("%w: unpaired surrogate", ErrUnsupported)
				}
				low, err := strconv.ParseUint(string(data[i+3:i+7]), 16, 16)
				if err != nil || low < 0xdc00 || low > 0xdfff {
					return fmt.Errorf("%w: unpaired surrogate", ErrUnsupported)
				}
				i += 6
			}
		}
	}
	return nil
}
