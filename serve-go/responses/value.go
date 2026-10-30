package responses

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"strata/requestnorm"
)

// ErrUnsupported identifies a request or event that this port cannot handle
// without silently dropping information or changing the Python result.
var ErrUnsupported = errors.New("responses: unsupported input")

// Error is a Responses API error. Param and Code are empty when Python would
// return null for them. Body produces the standard error envelope.
type Error struct {
	Message string
	Param   string
	Code    string
	Status  int
	Kind    string
}

func (e *Error) Error() string { return e.Message }
func (e *Error) Unwrap() error {
	if e.Code == "unsupported_parameter" {
		return ErrUnsupported
	}
	return nil
}

func (e *Error) Body() map[string]any {
	var param, code any
	if e.Param != "" {
		param = e.Param
	}
	if e.Code != "" {
		code = e.Code
	}
	kind := e.Kind
	if kind == "" {
		kind = "invalid_request_error"
	}
	return map[string]any{"error": map[string]any{
		"message": e.Message, "type": kind, "param": param, "code": code,
	}}
}

func invalid(path, message string) error {
	return &Error{Message: message, Param: path, Status: 400}
}

func unsupported(path, message string) error {
	return &Error{Message: message, Param: path, Code: "unsupported_parameter", Status: 400}
}

func field(v requestnorm.Value, name string) requestnorm.Value {
	out, _ := v.Lookup(name)
	return out
}

func stringOf(v requestnorm.Value) (string, bool) {
	if v.Kind() != requestnorm.String {
		return "", false
	}
	return v.Text()
}

func optionalString(v requestnorm.Value, path string) (string, error) {
	if v.Kind() == requestnorm.Null {
		return "", nil
	}
	s, ok := stringOf(v)
	if !ok {
		return "", unsupported(path, path+" must be a string or null")
	}
	return s, nil
}

func known(v requestnorm.Value, path string, names ...string) error {
	if v.Kind() != requestnorm.Object {
		return invalid(path, path+" must be an object")
	}
	allowed := make(map[string]bool, len(names))
	for _, name := range names {
		allowed[name] = true
	}
	for _, f := range v.Fields() {
		if !allowed[f.Name] {
			p := f.Name
			if path != "" {
				p = path + "." + p
			}
			return unsupported(p, "unsupported field "+p)
		}
	}
	return nil
}

func parse(data string) requestnorm.Value {
	v, err := requestnorm.ParseValue([]byte(data))
	if err != nil {
		panic(err) // only constants and encoding/json output reach this helper
	}
	return v
}

func jsonString(s string) string {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	_ = e.Encode(s)
	return strings.TrimSuffix(b.String(), "\n")
}

func strValue(s string) requestnorm.Value { return parse(jsonString(s)) }

func object(fields ...requestnorm.Field) requestnorm.Value {
	var b strings.Builder
	b.WriteByte('{')
	for i, f := range fields {
		if i != 0 {
			b.WriteByte(',')
		}
		b.WriteString(jsonString(f.Name))
		b.WriteByte(':')
		v, err := f.Value.MarshalJSON()
		if err != nil {
			panic(err)
		}
		b.Write(v)
	}
	b.WriteByte('}')
	return parse(b.String())
}

func f(name string, v requestnorm.Value) requestnorm.Field {
	return requestnorm.Field{Name: name, Value: v}
}

// pythonJSON implements the default json.dumps separators and Unicode mode
// used for tool outputs and unstreamed call arguments. Floats are refused:
// Go and Python do not always render the same binary64 value the same way.
func pythonJSON(v requestnorm.Value) (string, error) {
	var b strings.Builder
	var appendValue func(requestnorm.Value) error
	appendValue = func(v requestnorm.Value) error {
		switch v.Kind() {
		case requestnorm.Null:
			b.WriteString("null")
		case requestnorm.Boolean:
			x, _ := v.Bool()
			if x {
				b.WriteString("true")
			} else {
				b.WriteString("false")
			}
		case requestnorm.Number:
			x, ok := v.Integer()
			if !ok {
				return fmt.Errorf("%w: float in Python-formatted JSON", ErrUnsupported)
			}
			b.WriteString(x)
		case requestnorm.String:
			x, _ := v.Text()
			s := jsonString(x)
			// encoding/json escapes these two characters even with HTML escaping off.
			// Replace only actual escapes, not an escaped literal backslash.
			for i := 0; i < len(s); i++ {
				if s[i] == '\\' && i+1 < len(s) {
					if strings.HasPrefix(s[i:], `\u2028`) {
						b.WriteRune('\u2028')
						i += 5
						continue
					}
					if strings.HasPrefix(s[i:], `\u2029`) {
						b.WriteRune('\u2029')
						i += 5
						continue
					}
					b.WriteByte(s[i])
					i++
				}
				b.WriteByte(s[i])
			}
		case requestnorm.Array:
			b.WriteByte('[')
			for i, x := range v.Items() {
				if i > 0 {
					b.WriteString(", ")
				}
				if err := appendValue(x); err != nil {
					return err
				}
			}
			b.WriteByte(']')
		case requestnorm.Object:
			b.WriteByte('{')
			for i, x := range v.Fields() {
				if i > 0 {
					b.WriteString(", ")
				}
				if err := appendValue(strValue(x.Name)); err != nil {
					return err
				}
				b.WriteString(": ")
				if err := appendValue(x.Value); err != nil {
					return err
				}
			}
			b.WriteByte('}')
		default:
			return fmt.Errorf("%w: invalid JSON value", ErrUnsupported)
		}
		return nil
	}
	if err := appendValue(v); err != nil {
		return "", err
	}
	return b.String(), nil
}
