// Package requestnorm normalizes a strict subset of OpenAI Chat Completions
// requests to Strata's template messages without loading a model.
package requestnorm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"
)

type Family string

const (
	FamilyQwen     Family = "qwen"
	FamilyDeepSeek Family = "deepseek41"
)

// Normalizer owns the 64-entry tool-order cache. Reuse one per server to match
// the Python process-wide cache. Its methods are safe for concurrent use.
type Normalizer struct {
	family Family
	mu     sync.Mutex
	orders []toolOrder // oldest first
}

type toolOrder struct {
	key   string
	names []string
}

func NewNormalizer(family Family) (*Normalizer, error) {
	if family != FamilyQwen && family != FamilyDeepSeek {
		return nil, fmt.Errorf("%w: unknown model family %q", ErrUnsupported, family)
	}
	return &Normalizer{family: family}, nil
}

// Result contains exactly the inputs passed to the chat template. IDs remain
// available on Messages/Calls but are excluded from their JSON template form.
type Result struct {
	Messages []Message `json:"messages"`
	Tools    []Value   `json:"tools"`
	Kwargs   Value     `json:"kwargs"`
}

type Message struct {
	Role             string          `json:"role"`
	Content          TemplateContent `json:"content"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"-"`
}

type ToolCall struct {
	Function FunctionCall `json:"function"`
	ID       string       `json:"-"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments Value  `json:"arguments"`
}

// TemplateContent marshals as a string, or an ordered list of text/image
// objects if any image was present.
type TemplateContent struct {
	Text  string
	Parts []Value
}

func (c TemplateContent) MarshalJSON() ([]byte, error) {
	if c.Parts == nil {
		return stringValue(c.Text).MarshalJSON()
	}
	return json.Marshal(c.Parts)
}

// RequireMessages matches frontend.require_messages. NormalizeOpenAI does not
// call it, just as openai_to_messages does not.
func RequireMessages(messages []Message) error {
	if len(messages) == 0 {
		return errors.New("messages must be a non-empty list: send at least one message")
	}
	return nil
}

// NormalizeOpenAI converts a typed request. Unknown fields and unsupported
// content must be rejected by DecodeOpenAI before calling it for JSON input.
func (n *Normalizer) NormalizeOpenAI(req OpenAIRequest) (Result, error) {
	result := Result{Messages: make([]Message, 0, len(req.Messages)), Kwargs: objectValue()}
	for i, input := range req.Messages {
		if !utf8.ValidString(input.Role) || !utf8.ValidString(input.ReasoningContent) || !utf8.ValidString(input.ToolCallID) {
			return Result{}, fmt.Errorf("%w: invalid UTF-8 in messages[%d]", ErrUnsupported, i)
		}
		role := input.Role
		switch role {
		case "developer":
			role = "system"
		case "system", "user", "assistant", "tool":
		default:
			return Result{}, fmt.Errorf("%w: messages[%d].role %q", ErrUnsupported, i, role)
		}
		if role != "tool" && input.ToolCallID != "" {
			return Result{}, fmt.Errorf("%w: messages[%d].tool_call_id outside a tool message", ErrUnsupported, i)
		}
		content, err := normalizeContent(input.Content, role, i)
		if err != nil {
			return Result{}, err
		}
		message := Message{Role: role, Content: content, ReasoningContent: input.ReasoningContent,
			ToolCallID: input.ToolCallID}
		for _, call := range input.ToolCalls {
			if call.Name == "" || !utf8.ValidString(call.Name) || !utf8.ValidString(call.ID) {
				return Result{}, fmt.Errorf("%w: empty tool call name", ErrUnsupported)
			}
			args, err := toolArguments(call.Arguments)
			if err != nil {
				return Result{}, err
			}
			message.ToolCalls = append(message.ToolCalls, ToolCall{
				ID: call.ID, Function: FunctionCall{Name: call.Name, Arguments: args},
			})
		}
		result.Messages = append(result.Messages, message)
	}
	orderToolResults(result.Messages)
	for i := 1; i < len(result.Messages); i++ {
		if result.Messages[i].Role == "system" {
			result.Messages[i].Role = "user"
		}
	}
	tools, names, err := validateTools(req.Tools)
	if err != nil {
		return Result{}, err
	}
	result.Tools = n.stableTools(tools, names)
	effort := req.ReasoningEffort
	if isEmptyEffort(effort) {
		effort = req.ReasoningFallback
	}
	result.Kwargs, err = n.effort(effort)
	if err != nil {
		return Result{}, err
	}
	for _, option := range req.ChatTemplateKwargs {
		switch option.Name {
		case "enable_thinking":
			b, ok := option.Value.Bool()
			if !ok {
				return Result{}, fmt.Errorf("%w: chat_template_kwargs.enable_thinking must be a boolean", ErrUnsupported)
			}
			if !b {
				result.Kwargs = objectValue(Field{Name: "enable_thinking", Value: Value{kind: Boolean}})
			}
		case "reasoning_effort":
			if _, present := result.Kwargs.Lookup("enable_thinking"); !present {
				kw, err := n.effort(option.Value)
				if err != nil {
					return Result{}, err
				}
				for _, field := range kw.fields {
					result.Kwargs = setField(result.Kwargs, field)
				}
			}
		default:
			return Result{}, fmt.Errorf("%w: chat_template_kwargs.%s", ErrUnsupported, option.Name)
		}
	}
	return result, nil
}

func setField(v Value, field Field) Value {
	for i := range v.fields {
		if v.fields[i].Name == field.Name {
			v.fields[i] = field
			return v
		}
	}
	v.fields = append(v.fields, field)
	return v
}

func normalizeContent(input InputContent, role string, index int) (TemplateContent, error) {
	if !utf8.ValidString(input.Text) {
		return TemplateContent{}, fmt.Errorf("%w: invalid UTF-8 in messages[%d].content", ErrUnsupported, index)
	}
	if input.Parts == nil {
		return TemplateContent{Text: input.Text}, nil
	}
	if input.Text != "" {
		return TemplateContent{}, fmt.Errorf("%w: messages[%d].content has both text and parts", ErrUnsupported, index)
	}
	hasImage := false
	for _, part := range input.Parts {
		if !utf8.ValidString(part.Text) || !utf8.ValidString(part.Source) {
			return TemplateContent{}, fmt.Errorf("%w: invalid UTF-8 in messages[%d].content", ErrUnsupported, index)
		}
		switch part.Type {
		case "text":
			if part.Source != "" {
				return TemplateContent{}, fmt.Errorf("%w: source on text part", ErrUnsupported)
			}
		case "image":
			if part.HasText || part.Text != "" {
				return TemplateContent{}, fmt.Errorf("%w: text on image part", ErrUnsupported)
			}
			if role == "system" {
				return TemplateContent{}, fmt.Errorf("%w: image in system message", ErrUnsupported)
			}
			hasImage = true
		default:
			return TemplateContent{}, fmt.Errorf("%w: content part type %q", ErrUnsupported, part.Type)
		}
	}
	if !hasImage {
		var text strings.Builder
		for _, part := range input.Parts {
			text.WriteString(part.Text)
		}
		return TemplateContent{Text: text.String()}, nil
	}
	out := TemplateContent{Parts: make([]Value, 0, len(input.Parts))}
	for _, part := range input.Parts {
		if part.Type == "image" {
			out.Parts = append(out.Parts, objectValue(
				Field{Name: "type", Value: stringValue("image")},
				Field{Name: "source", Value: stringValue(part.Source)},
			))
		} else if part.HasText || part.Text != "" {
			out.Parts = append(out.Parts, objectValue(
				Field{Name: "type", Value: stringValue("text")},
				Field{Name: "text", Value: stringValue(part.Text)},
			))
		}
	}
	return out, nil
}

func toolArguments(v Value) (Value, error) {
	switch v.kind {
	case Null:
		return objectValue(), nil
	case Object:
		return v, nil
	case String:
		if strings.TrimSpace(v.text) == "" {
			return objectValue(), nil
		}
		parsed, err := ParseValue([]byte(v.text))
		if err == nil && parsed.kind == Object {
			return parsed, nil
		}
		if errors.Is(err, ErrUnsupported) || (err != nil && hasNonfinite(v.text)) {
			return Value{}, fmt.Errorf("%w: tool arguments contain unsupported JSON", ErrUnsupported)
		}
		return objectValue(Field{Name: "arguments", Value: v}), nil
	default:
		var b bytes.Buffer
		if err := v.appendJSON(&b, true); err != nil {
			return Value{}, err
		}
		return objectValue(Field{Name: "arguments", Value: stringValue(b.String())}), nil
	}
}

// Python's JSON decoder admits these non-standard constants. Do not mistake an
// object containing one for malformed text and silently change the arguments.
func hasNonfinite(s string) bool {
	inString, escape := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inString {
			if escape {
				escape = false
			} else if c == '\\' {
				escape = true
			} else if c == '"' {
				inString = false
			}
			continue
		}
		if c == '"' {
			inString = true
			continue
		}
		for _, word := range []string{"NaN", "Infinity", "-Infinity"} {
			if strings.HasPrefix(s[i:], word) &&
				(i == 0 || strings.ContainsRune(":,[ \t\r\n", rune(s[i-1]))) &&
				(i+len(word) == len(s) || strings.ContainsRune(",]} \t\r\n", rune(s[i+len(word)]))) {
				return true
			}
		}
	}
	return false
}

func orderToolResults(messages []Message) {
	for i := 0; i < len(messages); {
		if messages[i].Role != "tool" {
			i++
			continue
		}
		j := i + 1
		for j < len(messages) && messages[j].Role == "tool" {
			j++
		}
		if i > 0 && messages[i-1].Role == "assistant" && len(messages[i-1].ToolCalls) == j-i {
			rank := make(map[string]int, j-i)
			valid := true
			for k, call := range messages[i-1].ToolCalls {
				if _, exists := rank[call.ID]; exists || call.ID == "" {
					valid = false
				}
				rank[call.ID] = k
			}
			seen := make(map[string]bool, j-i)
			for _, message := range messages[i:j] {
				_, exists := rank[message.ToolCallID]
				if !exists || message.ToolCallID == "" || seen[message.ToolCallID] {
					valid = false
				}
				seen[message.ToolCallID] = true
			}
			if valid {
				sort.SliceStable(messages[i:j], func(a, b int) bool {
					return rank[messages[i+a].ToolCallID] < rank[messages[i+b].ToolCallID]
				})
			}
		}
		i = j
	}
}

func openAIToolShapeError() string {
	return `tools must be a list of tool objects, each {"type": "function", "function": {"name": ..., "parameters": {...}}}`
}

func validateTools(inputs []Value) ([]Value, []string, error) {
	var tools []Value
	var names []string
	shape := `{"type": "function", "function": {"name": ..., "parameters": {...}}}`
	for i, input := range inputs {
		if input.kind != Object {
			return nil, nil, errors.New(openAIToolShapeError())
		}
		tool := input
		typ := get(input, "type")
		if typ.kind != Null && (typ.kind != String || typ.text != "function") {
			return nil, nil, fmt.Errorf("%w: tools[%d].type", ErrUnsupported, i)
		}
		if typ.kind == String && typ.text == "function" {
			if wrapped, ok := input.Lookup("function"); ok {
				if err := knownFields(input, fmt.Sprintf("tools[%d]", i), "type", "function"); err != nil {
					return nil, nil, err
				}
				tool = wrapped
			}
		}
		name := get(tool, "name")
		if tool.kind != Object || name.kind != String || name.text == "" {
			return nil, nil, fmt.Errorf("tools[%d] has no name: each tool must be %s", i, shape)
		}
		schema := get(tool, "parameters")
		if schema.kind != Null && schema.kind != Object {
			return nil, nil, fmt.Errorf(`tools[%d] (%s): "parameters" must be an object (the JSON schema of its parameters), not %s`,
				i, name.text, pythonType(schema))
		}
		tools = append(tools, tool)
		names = append(names, name.text)
	}
	return tools, names, nil
}

func pythonType(v Value) string {
	switch v.kind {
	case Boolean:
		return "bool"
	case String:
		return "str"
	case Array:
		return "list"
	case Number:
		if v.isFloat {
			return "float"
		}
		return "int"
	default:
		return "NoneType"
	}
}

func (n *Normalizer) stableTools(tools []Value, names []string) []Value {
	if len(tools) < 2 {
		return tools
	}
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	for i := 1; i < len(sorted); i++ {
		if sorted[i-1] == sorted[i] {
			return tools
		}
	}
	keyBytes, _ := json.Marshal(sorted)
	key := string(keyBytes)
	n.mu.Lock()
	defer n.mu.Unlock()
	for i, order := range n.orders {
		if order.key == key {
			n.orders = append(append(n.orders[:i], n.orders[i+1:]...), order)
			byName := make(map[string]Value, len(tools))
			for j, name := range names {
				byName[name] = tools[j]
			}
			out := make([]Value, 0, len(tools))
			for _, name := range order.names {
				out = append(out, byName[name])
			}
			return out
		}
	}
	n.orders = append(n.orders, toolOrder{key: key, names: append([]string(nil), names...)})
	if len(n.orders) > 64 {
		n.orders = n.orders[1:]
	}
	return tools
}

func (n *Normalizer) effort(v Value) (Value, error) {
	empty := objectValue()
	if isEmptyEffort(v) {
		return empty, nil
	}
	if v.kind == Boolean && !v.flag {
		return objectValue(Field{Name: "enable_thinking", Value: v}), nil
	}
	if n.family == FamilyDeepSeek && v.kind == Number && !v.isFloat {
		num, err := strconv.Atoi(v.text)
		if err != nil || num < 1 || num > 100 {
			repr := v.text
			if repr == "-0" {
				repr = "0"
			}
			return Value{}, fmt.Errorf("unknown reasoning effort %s: use none, low, medium, high, max or 1..100", repr)
		}
		return objectValue(Field{Name: "reasoning_effort", Value: v}), nil
	}
	if v.kind != String {
		return Value{}, fmt.Errorf("%w: reasoning effort must be a string, false or a DeepSeek integer", ErrUnsupported)
	}
	var level string
	switch strings.ToLower(strings.TrimSpace(v.text)) {
	case "none", "off", "minimal", "disabled", "false":
		return objectValue(Field{Name: "enable_thinking", Value: Value{kind: Boolean}}), nil
	case "low":
		level = "low"
	case "medium":
		level = "medium"
	case "high":
		if n.family == FamilyDeepSeek {
			level = "high"
		} else {
			level = "xhigh"
		}
	case "xhigh", "max", "maximum":
		if n.family == FamilyDeepSeek {
			level = "max"
		} else {
			level = "xhigh"
		}
	default:
		suffix := ""
		if n.family == FamilyDeepSeek {
			suffix = ", max or 1..100"
		}
		return Value{}, fmt.Errorf("unknown reasoning effort %s: use none, low, medium or high%s", pythonQuote(v.text), suffix)
	}
	return objectValue(Field{Name: "reasoning_effort", Value: stringValue(level)}), nil
}
