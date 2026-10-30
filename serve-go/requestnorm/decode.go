package requestnorm

import (
	"errors"
	"fmt"
	"strings"
)

// OpenAIRequest is the supported prompt-normalization portion of a Chat
// Completions request. DecodeOpenAI also accepts double-encoded message,
// tool-call and tool arrays as the Python frontend does.
type OpenAIRequest struct {
	Messages           []OpenAIMessage
	Tools              []Value
	ReasoningEffort    Value
	ReasoningFallback  Value
	ChatTemplateKwargs []Field
}

// OpenAIMessage carries one conversation turn, including the IDs needed to
// associate a complete run of tool results with the preceding assistant calls.
type OpenAIMessage struct {
	Role             string
	Content          InputContent
	ReasoningContent string
	ToolCalls        []InputToolCall
	ToolCallID       string
}

// InputContent is either Text or Parts. A non-nil Parts slice selects the
// latter, even when it is empty. Null content is represented by empty Text.
type InputContent struct {
	Text  string
	Parts []InputPart
}

// InputPart is either text or an image. HasText distinguishes a missing text
// field from an explicitly empty text field in an image-containing content list.
type InputPart struct {
	Type    string // "text" or "image"
	Text    string
	HasText bool
	Source  string
}

type InputToolCall struct {
	ID        string
	Name      string
	Arguments Value
}

// DecodeOpenAI decodes a strict JSON request. Fields outside the documented
// normalization subset are rejected rather than ignored.
func DecodeOpenAI(data []byte) (OpenAIRequest, error) {
	root, err := ParseValue(data)
	if err != nil {
		return OpenAIRequest{}, err
	}
	if root.kind != Object {
		return OpenAIRequest{}, errors.New("request must be an object")
	}
	if err := knownFields(root, "request", "messages", "tools", "reasoning_effort", "reasoning", "chat_template_kwargs"); err != nil {
		return OpenAIRequest{}, err
	}
	var out OpenAIRequest
	messages, err := objectList(get(root, "messages"), "messages")
	if err != nil {
		return out, err
	}
	for i, msg := range messages {
		m, err := decodeMessage(msg, i)
		if err != nil {
			return OpenAIRequest{}, err
		}
		out.Messages = append(out.Messages, m)
	}
	tools := get(root, "tools")
	if tools.kind != Null && !(tools.kind == String && tools.text == "") &&
		!(tools.kind == Array && len(tools.items) == 0) {
		out.Tools, err = objectList(tools, "tools")
		if err != nil {
			if errors.Is(err, ErrUnsupported) {
				return OpenAIRequest{}, err
			}
			return OpenAIRequest{}, errors.New(openAIToolShapeError())
		}
	}
	out.ReasoningEffort = get(root, "reasoning_effort")
	reasoning := get(root, "reasoning")
	if reasoning.kind != Null {
		if reasoning.kind != Object {
			return OpenAIRequest{}, fmt.Errorf("%w: reasoning must be an object", ErrUnsupported)
		}
		if err := knownFields(reasoning, "reasoning", "effort"); err != nil {
			return OpenAIRequest{}, err
		}
		out.ReasoningFallback = get(reasoning, "effort")
	}
	options := get(root, "chat_template_kwargs")
	if options.kind != Null {
		if options.kind != Object {
			return OpenAIRequest{}, fmt.Errorf("%w: chat_template_kwargs must be an object", ErrUnsupported)
		}
		if err := knownFields(options, "chat_template_kwargs", "enable_thinking", "reasoning_effort"); err != nil {
			return OpenAIRequest{}, err
		}
		out.ChatTemplateKwargs = options.Fields()
	}
	return out, nil
}

func decodeMessage(v Value, i int) (OpenAIMessage, error) {
	path := fmt.Sprintf("messages[%d]", i)
	if err := knownFields(v, path, "role", "content", "reasoning_content", "tool_calls", "tool_call_id"); err != nil {
		return OpenAIMessage{}, err
	}
	role, ok := get(v, "role").Text()
	if !ok || get(v, "role").kind != String {
		return OpenAIMessage{}, fmt.Errorf("%w: %s.role must be a string", ErrUnsupported, path)
	}
	m := OpenAIMessage{Role: role}
	content, err := decodeContent(get(v, "content"), path+".content")
	if err != nil {
		return m, err
	}
	m.Content = content
	if m.ReasoningContent, err = optionalString(get(v, "reasoning_content"), path+".reasoning_content"); err != nil {
		return m, err
	}
	if m.ToolCallID, err = optionalString(get(v, "tool_call_id"), path+".tool_call_id"); err != nil {
		return m, err
	}
	calls := get(v, "tool_calls")
	if calls.kind == Null || (calls.kind == String && calls.text == "") || (calls.kind == Array && len(calls.items) == 0) {
		return m, nil
	}
	list, err := objectList(calls, "tool_calls")
	if err != nil {
		return m, err
	}
	for j, call := range list {
		path := fmt.Sprintf("%s.tool_calls[%d]", path, j)
		fn := call
		if wrapped, present := call.Lookup("function"); present {
			if wrapped.kind != Object {
				return m, errors.New(`tool_calls must be a list of objects (each with a "function" object)`)
			}
			if err := knownFields(call, path, "id", "type", "function"); err != nil {
				return m, err
			}
			fn = wrapped
			if err := knownFields(fn, path+".function", "name", "arguments"); err != nil {
				return m, err
			}
		} else if err := knownFields(call, path, "id", "type", "name", "arguments"); err != nil {
			return m, err
		}
		typ, err := optionalString(get(call, "type"), path+".type")
		if err != nil {
			return m, err
		}
		if typ != "" && typ != "function" {
			return m, fmt.Errorf("%w: %s.type %q", ErrUnsupported, path, typ)
		}
		id, err := optionalString(get(call, "id"), path+".id")
		if err != nil {
			return m, err
		}
		name, err := optionalString(get(fn, "name"), path+".name")
		if err != nil {
			return m, err
		}
		if name == "" {
			return m, fmt.Errorf("%w: %s.name must be non-empty", ErrUnsupported, path)
		}
		m.ToolCalls = append(m.ToolCalls, InputToolCall{ID: id, Name: name, Arguments: get(fn, "arguments")})
	}
	return m, nil
}

func decodeContent(v Value, path string) (InputContent, error) {
	switch v.kind {
	case Null:
		return InputContent{}, nil
	case String:
		return InputContent{Text: v.text}, nil
	case Array:
		content := InputContent{Parts: make([]InputPart, 0, len(v.items))}
		for i, item := range v.items {
			p, err := decodePart(item, fmt.Sprintf("%s[%d]", path, i))
			if err != nil {
				return InputContent{}, err
			}
			content.Parts = append(content.Parts, p)
		}
		return content, nil
	default:
		return InputContent{}, fmt.Errorf("%w: %s must be text, null or an array of parts", ErrUnsupported, path)
	}
}

func decodePart(v Value, path string) (InputPart, error) {
	if v.kind != Object {
		return InputPart{}, fmt.Errorf("%w: %s must be an object", ErrUnsupported, path)
	}
	if typ, present := v.Lookup("type"); present && typ.kind == String && typ.text == "" {
		return InputPart{}, fmt.Errorf("%w: %s.type %q", ErrUnsupported, path, typ.text)
	}
	typ, err := optionalString(get(v, "type"), path+".type")
	if err != nil {
		return InputPart{}, err
	}
	switch typ {
	case "", "text", "input_text":
		if err := knownFields(v, path, "type", "text"); err != nil {
			return InputPart{}, err
		}
		value, present := v.Lookup("text")
		text, err := optionalString(value, path+".text")
		if err != nil || (present && value.kind == Null) {
			return InputPart{}, fmt.Errorf("%w: %s.text must be a string", ErrUnsupported, path)
		}
		return InputPart{Type: "text", Text: text, HasText: present}, nil
	case "refusal":
		if err := knownFields(v, path, "type", "refusal"); err != nil {
			return InputPart{}, err
		}
		text, err := optionalString(get(v, "refusal"), path+".refusal")
		return InputPart{Type: "text", Text: text, HasText: true}, err
	case "image_url", "input_image":
		if err := knownFields(v, path, "type", "image_url"); err != nil {
			return InputPart{}, err
		}
		url := get(v, "image_url")
		if url.kind == Object {
			if err := knownFields(url, path+".image_url", "url"); err != nil {
				return InputPart{}, err
			}
			url = get(url, "url")
		}
		source, err := optionalString(url, path+".image_url")
		return InputPart{Type: "image", Source: source}, err
	case "image":
		if err := knownFields(v, path, "type", "source"); err != nil {
			return InputPart{}, err
		}
		source := get(v, "source")
		if source.kind == Null {
			return InputPart{Type: "image"}, nil
		}
		if source.kind != Object {
			return InputPart{}, fmt.Errorf("%w: %s.source must be an object", ErrUnsupported, path)
		}
		if err := knownFields(source, path+".source", "type", "media_type", "data", "url", "path"); err != nil {
			return InputPart{}, err
		}
		st, err := optionalString(get(source, "type"), path+".source.type")
		if err != nil {
			return InputPart{}, err
		}
		switch st {
		case "base64":
			if _, ok := source.Lookup("url"); ok {
				return InputPart{}, fmt.Errorf("%w: %s.source.url with base64 source", ErrUnsupported, path)
			}
			if _, ok := source.Lookup("path"); ok {
				return InputPart{}, fmt.Errorf("%w: %s.source.path with base64 source", ErrUnsupported, path)
			}
			media, exists := source.Lookup("media_type")
			mt := "image/png"
			if exists {
				mt, err = optionalString(media, path+".source.media_type")
				if err != nil || media.kind == Null {
					return InputPart{}, fmt.Errorf("%w: %s.source.media_type must be a string", ErrUnsupported, path)
				}
			}
			dataValue, exists := source.Lookup("data")
			if exists && dataValue.kind == Null {
				return InputPart{}, fmt.Errorf("%w: %s.source.data must be a string", ErrUnsupported, path)
			}
			data, err := optionalString(dataValue, path+".source.data")
			return InputPart{Type: "image", Source: "data:" + mt + ";base64," + data}, err
		case "", "url", "path":
			for _, field := range []string{"media_type", "data"} {
				if _, ok := source.Lookup(field); ok {
					return InputPart{}, fmt.Errorf("%w: %s.source.%s without base64 source", ErrUnsupported, path, field)
				}
			}
			url, err := optionalString(get(source, "url"), path+".source.url")
			if err != nil {
				return InputPart{}, err
			}
			fallback, err := optionalString(get(source, "path"), path+".source.path")
			if err != nil {
				return InputPart{}, err
			}
			if url != "" && fallback != "" {
				return InputPart{}, fmt.Errorf("%w: %s.source has both url and path", ErrUnsupported, path)
			}
			if url == "" {
				url = fallback
			}
			return InputPart{Type: "image", Source: url}, nil
		default:
			return InputPart{}, fmt.Errorf("%w: %s.source.type %q", ErrUnsupported, path, st)
		}
	default:
		return InputPart{}, fmt.Errorf("%w: %s.type %q", ErrUnsupported, path, typ)
	}
}

func objectList(v Value, name string) ([]Value, error) {
	if v.kind == Null {
		return nil, nil
	}
	if v.kind == String {
		var err error
		v, err = ParseValue([]byte(v.text))
		if err != nil {
			if errors.Is(err, ErrUnsupported) {
				return nil, err
			}
			return nil, fmt.Errorf("%s must be a list of objects (a string was sent that is not JSON)", name)
		}
	}
	if v.kind != Array {
		return nil, fmt.Errorf("%s must be a list of objects", name)
	}
	for _, item := range v.items {
		if item.kind != Object {
			return nil, fmt.Errorf("%s must be a list of objects", name)
		}
	}
	return v.Items(), nil
}

func optionalString(v Value, path string) (string, error) {
	if v.kind == Null {
		return "", nil
	}
	if v.kind != String {
		return "", fmt.Errorf("%w: %s must be a string", ErrUnsupported, path)
	}
	return v.text, nil
}

func get(v Value, name string) Value {
	value, _ := v.Lookup(name)
	return value
}

func knownFields(v Value, path string, allowed ...string) error {
	for _, field := range v.fields {
		found := false
		for _, name := range allowed {
			if field.Name == name {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%w: %s.%s", ErrUnsupported, path, field.Name)
		}
	}
	return nil
}

func isEmptyEffort(v Value) bool {
	return v.kind == Null || (v.kind == String && v.text == "")
}

func pythonQuote(s string) string {
	quote := "'"
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		quote = `"`
	}
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, quote, `\`+quote)
	s = strings.ReplaceAll(s, "\n", `\n`)
	s = strings.ReplaceAll(s, "\r", `\r`)
	s = strings.ReplaceAll(s, "\t", `\t`)
	return quote + s + quote
}
