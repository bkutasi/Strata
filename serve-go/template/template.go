// Package template is an experimental, opt-in model chat template renderer.
// It is not connected to Strata's server or its default Python renderer.
package template

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"
	"unicode"

	gj "github.com/jryberg/gojinja"
	"github.com/jryberg/gojinja/pkg/environment"
	"github.com/jryberg/gojinja/pkg/escape"
	"github.com/jryberg/gojinja/pkg/lexer"
	"github.com/jryberg/gojinja/pkg/runtime"
	"strata/requestnorm"
)

// ErrUnsupported means this experimental renderer cannot safely represent the
// requested operation. Callers must not treat it as a rendered prompt.
var ErrUnsupported = errors.New("template: unsupported")

// Renderer wraps the upstream Go Jinja renderer.
type Renderer struct{ template *gj.Template }

func New(source string) (r *Renderer, err error) {
	defer recoverError(&err)
	opts := lexer.DefaultOptions()
	opts.TrimBlocks, opts.LstripBlocks = true, true
	env, err := gj.New(
		gj.WithAutoescape(gj.AutoescapeNever{}),
		gj.WithLexerOptions(opts),
		gj.WithGlobal("raise_exception", func(message string) (string, error) { return "", &RequestError{Message: message} }),
		gj.WithFilter("tojson", environment.Filter{Func: toJSONFilter}),
	)
	if err != nil {
		return nil, err
	}
	t, err := env.FromString(source)
	if err != nil {
		return nil, err
	}
	return &Renderer{template: t}, nil
}

// RequestError is a request rejected by the model template.
type RequestError struct{ Message string }

func (e *RequestError) Error() string { return e.Message }

func recoverError(err *error) {
	if v := recover(); v != nil {
		*err = fmt.Errorf("template: renderer panic: %v", v)
	}
}

// Input contains the JSON forms of normalized messages, tools, and keyword
// arguments. JSON objects retain insertion order; numbers retain int/float
// types. Literal-token protection is not implemented here; callers must apply
// the marking and plain-span logic before tokenizing a rendered prompt.
type Input struct {
	Messages            []byte
	Tools               []byte
	Kwargs              []byte
	AddGenerationPrompt bool
	KeepEmptyTurns      bool
}

// Render renders the supplied variables. Cancellation is cooperative, not a
// memory limit. The upstream sandbox is not an immutable sandbox.
func (r *Renderer) Render(ctx context.Context, input Input) (out string, err error) {
	defer recoverError(&err)
	if r == nil || r.template == nil || ctx == nil {
		return "", fmt.Errorf("%w: missing renderer or context", ErrUnsupported)
	}
	if len(input.Tools) == 0 {
		input.Tools = []byte("null")
	}
	if len(input.Kwargs) == 0 {
		input.Kwargs = []byte("{}")
	}
	messages, err := decode(input.Messages)
	if err != nil {
		return "", err
	}
	tools, err := decode(input.Tools)
	if err != nil {
		return "", err
	}
	kwargs, err := decode(input.Kwargs)
	if err != nil {
		return "", err
	}
	if messages.Kind() != requestnorm.Array || (tools.Kind() != requestnorm.Array && tools.Kind() != requestnorm.Null) || kwargs.Kind() != requestnorm.Object {
		return "", fmt.Errorf("%w: expected messages array, tools array or null, and kwargs object", ErrUnsupported)
	}
	if _, ok := kwargs.Lookup("messages"); ok {
		return "", fmt.Errorf("%w: reserved keyword messages", ErrUnsupported)
	}
	if _, ok := kwargs.Lookup("tools"); ok {
		return "", fmt.Errorf("%w: reserved keyword tools", ErrUnsupported)
	}
	if _, ok := kwargs.Lookup("add_generation_prompt"); ok {
		return "", fmt.Errorf("%w: reserved keyword add_generation_prompt", ErrUnsupported)
	}
	m, err := encodeValue(messages)
	if err != nil {
		return "", err
	}
	list := m.([]any)
	if !input.KeepEmptyTurns {
		filtered := make([]any, 0, len(list))
		for i, item := range list {
			if i == len(list)-1 || !emptyAssistant(item) {
				filtered = append(filtered, item)
			}
		}
		list = filtered
	}
	t, err := encodeValue(tools)
	if err != nil {
		return "", err
	}
	k, err := encodeValue(kwargs)
	if err != nil {
		return "", err
	}
	vars := k.(*runtime.OrderedDict)
	vars.Set("messages", list)
	vars.Set("tools", t)
	vars.Set("add_generation_prompt", input.AddGenerationPrompt)
	out, err = r.template.RenderContext(ctx, vars)
	if err != nil {
		return "", err
	}
	return out, nil
}

func decode(data []byte) (requestnorm.Value, error) {
	v, err := requestnorm.ParseValue(data)
	if err != nil {
		return v, fmt.Errorf("%w: invalid JSON: %v", ErrUnsupported, err)
	}
	return v, nil
}

func encodeValue(v requestnorm.Value) (any, error) {
	switch v.Kind() {
	case requestnorm.Null:
		return nil, nil
	case requestnorm.Boolean:
		b, _ := v.Bool()
		return b, nil
	case requestnorm.String:
		s, _ := v.Text()
		return s, nil
	case requestnorm.Number:
		if f, ok := v.Float64(); ok {
			return f, nil
		}
		s, _ := v.Integer()
		n, ok := new(big.Int).SetString(s, 10)
		if !ok {
			return nil, fmt.Errorf("%w: invalid integer", ErrUnsupported)
		}
		if n.IsInt64() {
			return n.Int64(), nil
		}
		return n, nil
	case requestnorm.Array:
		items := v.Items()
		out := make([]any, 0, len(items))
		for _, item := range items {
			x, err := encodeValue(item)
			if err != nil {
				return nil, err
			}
			out = append(out, x)
		}
		return out, nil
	case requestnorm.Object:
		out := runtime.NewOrderedDict()
		for _, field := range v.Fields() {
			x, err := encodeValue(field.Value)
			if err != nil {
				return nil, err
			}
			out.Set(field.Name, x)
		}
		return out, nil
	}
	return nil, ErrUnsupported
}

func emptyAssistant(v any) bool {
	m, ok := v.(*runtime.OrderedDict)
	if !ok {
		return false
	}
	role, _ := m.Get("role")
	if role != "assistant" {
		return false
	}
	calls, _ := m.Get("tool_calls")
	if calls != nil {
		if list, ok := calls.([]any); !ok || len(list) > 0 {
			return false
		}
	}
	content, _ := m.Get("content")
	switch x := content.(type) {
	case nil:
		return true
	case string:
		return strings.TrimFunc(x, unicode.IsSpace) == ""
	case []any:
		var text strings.Builder
		for _, part := range x {
			d, ok := part.(*runtime.OrderedDict)
			if !ok {
				continue
			}
			kind, _ := d.Get("type")
			if kind == "image" || kind == "image_url" || kind == "input_image" {
				return false
			}
			if kind == nil || kind == "text" || kind == "input_text" {
				value, _ := d.Get("text")
				if s, ok := value.(string); ok {
					text.WriteString(s)
				}
			}
		}
		return strings.TrimFunc(text.String(), unicode.IsSpace) == ""
	default:
		return false
	}
}

func toJSONFilter(_ *environment.Environment, _ *runtime.Context, value any, args []any, kwargs map[string]any) (any, error) {
	if len(args) != 0 || len(kwargs) != 0 {
		return nil, fmt.Errorf("%w: tojson options", ErrUnsupported)
	}
	return pythonJSON(value)
}

func pythonJSON(v any) (string, error) {
	switch x := v.(type) {
	case nil:
		return "null", nil
	case bool:
		if x {
			return "true", nil
		}
		return "false", nil
	case int, int64, *big.Int:
		return fmt.Sprint(x), nil
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return "", fmt.Errorf("%w: non-finite JSON float", ErrUnsupported)
		}
		return escape.SoftStr(x), nil
	case string:
		var b bytes.Buffer
		e := json.NewEncoder(&b)
		e.SetEscapeHTML(false)
		if err := e.Encode(x); err != nil {
			return "", err
		}
		s := strings.TrimSuffix(b.String(), "\n")
		var out strings.Builder
		for i := 0; i < len(s); i++ {
			if s[i] == '\\' && i+1 < len(s) {
				if strings.HasPrefix(s[i:], `\u2028`) || strings.HasPrefix(s[i:], `\u2029`) {
					if s[i+5] == '8' {
						out.WriteRune('\u2028')
					} else {
						out.WriteRune('\u2029')
					}
					i += 5
					continue
				}
				out.WriteByte(s[i])
				i++
			}
			out.WriteByte(s[i])
		}
		return out.String(), nil
	case *runtime.OrderedDict:
		parts := make([]string, 0, x.Len())
		for _, key := range x.Keys() {
			k, ok := key.(string)
			if !ok {
				return "", fmt.Errorf("%w: non-string JSON key", ErrUnsupported)
			}
			a, _ := pythonJSON(k)
			value, _ := x.Get(key)
			b, err := pythonJSON(value)
			if err != nil {
				return "", err
			}
			parts = append(parts, a+": "+b)
		}
		return "{" + strings.Join(parts, ", ") + "}", nil
	case *runtime.PyList:
		return pythonJSON(x.Items())
	case runtime.Tuple:
		return pythonJSON([]any(x))
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			s, err := pythonJSON(item)
			if err != nil {
				return "", err
			}
			parts = append(parts, s)
		}
		return "[" + strings.Join(parts, ", ") + "]", nil
	default:
		return "", fmt.Errorf("%w: JSON value %T", ErrUnsupported, v)
	}
}
