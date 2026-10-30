package requestnorm

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

// AnthropicRequest is the supported prompt-normalization portion of a Messages
// request. Values retain JSON object order, notably in tool schemas and input.
// Construct values with ParseValue, or use DecodeAnthropic.
type AnthropicRequest struct {
	System                Value
	Messages              []Value
	Tools                 []Value
	Thinking              Value
	OutputConfig          Value
	ReasoningBudgetTokens Value
}

// AnthropicMessage embeds the shared Message, retaining call and result IDs
// outside its template JSON. HasReasoningContent distinguishes an explicitly
// empty thinking block from an absent reasoning_content field.
type AnthropicMessage struct {
	Message
	HasReasoningContent bool `json:"-"`
}

// MarshalJSON emits the Python template shape, including empty reasoning when
// a thinking block was present, and excluding the retained tool IDs.
func (m AnthropicMessage) MarshalJSON() ([]byte, error) {
	var reasoning *string
	if m.HasReasoningContent {
		reasoning = &m.ReasoningContent
	}
	return json.Marshal(struct {
		Role             string          `json:"role"`
		Content          TemplateContent `json:"content"`
		ReasoningContent *string         `json:"reasoning_content,omitempty"`
		ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
	}{m.Role, m.Content, reasoning, m.ToolCalls})
}

// AnthropicResult holds template inputs and the effective request budget.
// Python writes a positive integer thinking.budget_tokens into the request's
// reasoning_budget_tokens when the latter is absent or null. The separate
// ReasoningBudgetTokens field exposes that side effect without mutating input;
// it is not part of the template JSON. Callers must still validate and apply
// the budget as the server does, including its configuration fallback.
type AnthropicResult struct {
	Messages              []AnthropicMessage `json:"messages"`
	Tools                 []Value            `json:"tools"`
	Kwargs                Value              `json:"kwargs"`
	ReasoningBudgetTokens Value              `json:"-"`
}

// RequireAnthropicMessages matches the separate Python require_messages check.
func RequireAnthropicMessages(messages []AnthropicMessage) error {
	if len(messages) == 0 {
		return RequireMessages(nil)
	}
	return nil
}

// DecodeAnthropic decodes the strict subset consumed by NormalizeAnthropic.
// It accepts double-encoded messages and tools, but rejects every other
// top-level field, including model, sampling, streaming and tool_choice.
// Unsupported blocks and metadata are rejected during normalization; this is
// not a complete Messages HTTP handler.
func DecodeAnthropic(data []byte) (AnthropicRequest, error) {
	root, err := ParseValue(data)
	if err != nil {
		return AnthropicRequest{}, err
	}
	if root.kind != Object {
		return AnthropicRequest{}, errors.New("request must be an object")
	}
	if err := knownFields(root, "request", "system", "messages", "tools", "thinking", "output_config",
		"reasoning_budget_tokens"); err != nil {
		return AnthropicRequest{}, err
	}
	messages, err := objectList(get(root, "messages"), "messages")
	if err != nil {
		return AnthropicRequest{}, err
	}
	var tools []Value
	rawTools := get(root, "tools")
	if rawTools.kind != Null && !(rawTools.kind == String && rawTools.text == "") &&
		!(rawTools.kind == Array && len(rawTools.items) == 0) {
		tools, err = objectList(rawTools, "tools")
		if err != nil {
			if errors.Is(err, ErrUnsupported) {
				return AnthropicRequest{}, err
			}
			return AnthropicRequest{}, errors.New(anthropicToolShapeError())
		}
	}
	return AnthropicRequest{
		System: get(root, "system"), Messages: messages, Tools: tools,
		Thinking: get(root, "thinking"), OutputConfig: get(root, "output_config"),
		ReasoningBudgetTokens: get(root, "reasoning_budget_tokens"),
	}, nil
}

// NormalizeAnthropic converts Messages history into template inputs. Pass true
// for Python's default think_unasked behavior, or false for "on_request".
// It preserves the received order of tool results, as Python's Anthropic path
// does, even when their IDs do not match call order. It shares the normalizer's
// tool-order cache with NormalizeOpenAI.
//
// Only text, refusal, images, thinking, tool_use and tool_result are supported.
// Unknown fields, redacted thinking, server tools, cache-control metadata,
// unsupported/lossy values, and user-image turns whose other blocks Python
// would discard return ErrUnsupported. Top-level system images are rejected.
// ENV_TO_USER, tool choice, shared server configuration, image fetching,
// template rendering and server-side request/budget validation are separate.
// Value's documented Python JSON serialization limitations also apply.
func (n *Normalizer) NormalizeAnthropic(req AnthropicRequest, thinkUnasked bool) (AnthropicResult, error) {
	result := AnthropicResult{
		Messages: make([]AnthropicMessage, 0, len(req.Messages)+1),
		Kwargs:   objectValue(), ReasoningBudgetTokens: req.ReasoningBudgetTokens,
	}
	system, present, err := anthropicSystem(req.System)
	if err != nil {
		return AnthropicResult{}, err
	}
	if present {
		result.Messages = append(result.Messages, AnthropicMessage{Message: Message{
			Role: "system", Content: TemplateContent{Text: pinBillingStamp(system)},
		}})
	}
	for i, input := range req.Messages {
		messages, err := anthropicMessages(input, i)
		if err != nil {
			return AnthropicResult{}, err
		}
		result.Messages = append(result.Messages, messages...)
	}
	for i := 1; i < len(result.Messages); i++ {
		if result.Messages[i].Role == "system" {
			result.Messages[i].Role = "user"
		}
	}
	tools, names, err := anthropicTools(req.Tools)
	if err != nil {
		return AnthropicResult{}, err
	}
	result.Tools = n.stableTools(tools, names)
	result.Kwargs, result.ReasoningBudgetTokens, err = n.anthropicThinking(
		req.Thinking, req.OutputConfig, req.ReasoningBudgetTokens, thinkUnasked)
	if err != nil {
		return AnthropicResult{}, err
	}
	return result, nil
}

func anthropicSystem(system Value) (string, bool, error) {
	switch system.kind {
	case Null:
		return "", false, nil
	case String:
		return system.text, system.text != "", nil
	case Array:
		if len(system.items) == 0 {
			return "", false, nil
		}
		content, err := anthropicContent(system, "system")
		if err != nil {
			return "", false, err
		}
		normalized, err := normalizeContent(content, "system", 0)
		return normalized.Text, true, err
	default:
		return "", false, fmt.Errorf("%w: system must be text, null or an array of text parts", ErrUnsupported)
	}
}

// The shared OpenAI decoder treats type:"" like a missing type. Python's
// Anthropic readers do not: _text_of and _parts_of ignore the former.
func anthropicContent(value Value, path string) (InputContent, error) {
	if value.kind == Array {
		for i, part := range value.items {
			typ, present := part.Lookup("type")
			if present && typ.kind == String && typ.text == "" {
				return InputContent{}, fmt.Errorf("%w: %s[%d].type is not read", ErrUnsupported, path, i)
			}
		}
	}
	return decodeContent(value, path)
}

func anthropicMessages(input Value, index int) ([]AnthropicMessage, error) {
	path := fmt.Sprintf("messages[%d]", index)
	if input.kind != Object {
		return nil, fmt.Errorf("%w: %s must be an object", ErrUnsupported, path)
	}
	if err := knownFields(input, path, "role", "content"); err != nil {
		return nil, err
	}
	role, ok := get(input, "role").Text()
	if !ok || get(input, "role").kind != String {
		return nil, fmt.Errorf("%w: %s.role must be a string", ErrUnsupported, path)
	}
	switch role {
	case "user", "assistant", "system", "tool":
	default:
		return nil, fmt.Errorf("%w: %s.role %q", ErrUnsupported, path, role)
	}
	content := get(input, "content")
	switch content.kind {
	case String:
		return []AnthropicMessage{{Message: Message{Role: role, Content: TemplateContent{Text: content.text}}}}, nil
	case Null:
		return nil, nil
	case Array:
	default:
		return nil, fmt.Errorf("%w: %s.content must be text, null or an array of blocks", ErrUnsupported, path)
	}

	hasImage, hasResult := false, false
	for _, block := range content.items {
		switch get(block, "type").text {
		case "image", "image_url", "input_image":
			hasImage = true
		case "tool_result":
			hasResult = true
		}
	}
	fastImage := role == "user" && hasImage && !hasResult
	var out []AnthropicMessage
	parts := InputContent{Parts: make([]InputPart, 0, len(content.items))}
	var reasoning []string
	var calls []ToolCall
	for j, block := range content.items {
		blockPath := fmt.Sprintf("%s.content[%d]", path, j)
		if block.kind != Object {
			return nil, fmt.Errorf("%w: %s must be an object", ErrUnsupported, blockPath)
		}
		typ := get(block, "type")
		if typ.kind != String && typ.kind != Null {
			return nil, fmt.Errorf("%w: %s.type must be a string", ErrUnsupported, blockPath)
		}
		switch typ.text {
		case "text", "refusal", "image", "image_url", "input_image":
			part, err := decodePart(block, blockPath)
			if err != nil {
				return nil, err
			}
			parts.Parts = append(parts.Parts, part)
		case "", "input_text":
			// _parts_of reads these only on the user-image fast path.
			if !fastImage || (typ.kind == String && typ.text == "") {
				return nil, fmt.Errorf("%w: %s.type is not read in this turn", ErrUnsupported, blockPath)
			}
			part, err := decodePart(block, blockPath)
			if err != nil {
				return nil, err
			}
			parts.Parts = append(parts.Parts, part)
		case "thinking":
			if fastImage {
				return nil, fmt.Errorf("%w: %s would be discarded from an image turn", ErrUnsupported, blockPath)
			}
			if err := knownFields(block, blockPath, "type", "thinking"); err != nil {
				return nil, err
			}
			thinking, present := block.Lookup("thinking")
			if present && thinking.kind != String {
				return nil, fmt.Errorf("%w: %s.thinking must be a string", ErrUnsupported, blockPath)
			}
			reasoning = append(reasoning, thinking.text)
		case "tool_use":
			if fastImage {
				return nil, fmt.Errorf("%w: %s would be discarded from an image turn", ErrUnsupported, blockPath)
			}
			if err := knownFields(block, blockPath, "type", "id", "name", "input"); err != nil {
				return nil, err
			}
			id, err := optionalString(get(block, "id"), blockPath+".id")
			if err != nil {
				return nil, err
			}
			name := get(block, "name")
			if name.kind != String || name.text == "" {
				return nil, fmt.Errorf("%w: %s.name must be a non-empty string", ErrUnsupported, blockPath)
			}
			args, err := toolArguments(get(block, "input"))
			if err != nil {
				return nil, err
			}
			calls = append(calls, ToolCall{ID: id, Function: FunctionCall{Name: name.text, Arguments: args}})
		case "tool_result":
			if err := knownFields(block, blockPath, "type", "tool_use_id", "content"); err != nil {
				return nil, err
			}
			id, err := optionalString(get(block, "tool_use_id"), blockPath+".tool_use_id")
			if err != nil {
				return nil, err
			}
			content, err := anthropicContent(get(block, "content"), blockPath+".content")
			if err != nil {
				return nil, err
			}
			body, err := normalizeContent(content, "tool", index)
			if err != nil {
				return nil, err
			}
			out = append(out, AnthropicMessage{Message: Message{Role: "tool", Content: body, ToolCallID: id}})
		default:
			return nil, fmt.Errorf("%w: %s.type %q", ErrUnsupported, blockPath, typ.text)
		}
	}
	if len(parts.Parts) != 0 || len(reasoning) != 0 || len(calls) != 0 {
		// Anthropic's block path renders images in any supported message role.
		body, err := normalizeContent(parts, "user", index)
		if err != nil {
			return nil, err
		}
		out = append(out, AnthropicMessage{
			Message:             Message{Role: role, Content: body, ReasoningContent: strings.Join(reasoning, ""), ToolCalls: calls},
			HasReasoningContent: len(reasoning) != 0,
		})
	}
	return out, nil
}

func anthropicToolShapeError() string {
	return `tools must be a list of tool objects, each {"name": ..., "input_schema": {...}}`
}

func anthropicTools(inputs []Value) ([]Value, []string, error) {
	var tools []Value
	var names []string
	for i, input := range inputs {
		path := fmt.Sprintf("tools[%d]", i)
		if input.kind != Object {
			return nil, nil, errors.New(anthropicToolShapeError())
		}
		if err := knownFields(input, path, "name", "description", "input_schema"); err != nil {
			return nil, nil, err
		}
		name := get(input, "name")
		if name.kind != String || name.text == "" {
			return nil, nil, fmt.Errorf("%s has no name: each tool must be %s", path,
				`{"name": ..., "input_schema": {...}}`)
		}
		description, present := input.Lookup("description")
		if !present {
			description = stringValue("")
		} else if description.kind != String && description.kind != Null {
			return nil, nil, fmt.Errorf("%w: %s.description must be a string or null", ErrUnsupported, path)
		}
		schema, present := input.Lookup("input_schema")
		if !present {
			schema = objectValue()
		} else if schema.kind != Object && schema.kind != Null {
			return nil, nil, fmt.Errorf(`%s (%s): "input_schema" must be an object (the JSON schema of its parameters), not %s`,
				path, name.text, pythonType(schema))
		}
		tools = append(tools, objectValue(
			Field{Name: "name", Value: name},
			Field{Name: "description", Value: description},
			Field{Name: "parameters", Value: schema},
		))
		names = append(names, name.text)
	}
	return tools, names, nil
}

func (n *Normalizer) anthropicThinking(thinking, config, explicitBudget Value, thinkUnasked bool) (Value, Value, error) {
	empty := objectValue()
	if explicitBudget.kind != Null && explicitBudget.kind != Number {
		return Value{}, Value{}, fmt.Errorf("%w: reasoning_budget_tokens must be a number or null", ErrUnsupported)
	}
	if thinking.kind != Null && thinking.kind != Object {
		return Value{}, Value{}, fmt.Errorf("%w: thinking must be an object or null", ErrUnsupported)
	}
	if err := knownFields(thinking, "thinking", "type", "budget_tokens"); err != nil {
		return Value{}, Value{}, err
	}
	typ := get(thinking, "type")
	if typ.kind != Null && (typ.kind != String || (typ.text != "enabled" && typ.text != "disabled" && typ.text != "adaptive")) {
		return Value{}, Value{}, fmt.Errorf("%w: thinking.type", ErrUnsupported)
	}
	if config.kind != Null && config.kind != Object {
		return Value{}, Value{}, fmt.Errorf("%w: output_config must be an object or null", ErrUnsupported)
	}
	if err := knownFields(config, "output_config", "effort"); err != nil {
		return Value{}, Value{}, err
	}
	effort := get(config, "effort")
	if effort.kind != Null && effort.kind != String && effort.kind != Boolean && effort.kind != Number {
		return Value{}, Value{}, fmt.Errorf("%w: output_config.effort", ErrUnsupported)
	}
	budget := get(thinking, "budget_tokens")
	if budget.kind != Null && budget.kind != String && budget.kind != Number && budget.kind != Boolean {
		return Value{}, Value{}, fmt.Errorf("%w: thinking.budget_tokens", ErrUnsupported)
	}
	if typ.text != "disabled" && budget.kind == Number && !strings.ContainsAny(budget.text, ".eE") {
		value, ok := new(big.Int).SetString(budget.text, 10)
		if ok && value.Sign() > 0 && explicitBudget.kind == Null {
			explicitBudget = budget
		}
	}
	if typ.text == "disabled" {
		return objectValue(Field{Name: "enable_thinking", Value: Value{kind: Boolean}}), explicitBudget, nil
	}
	if anthropicTruthy(effort) {
		kwargs, err := n.effort(effort)
		return kwargs, explicitBudget, err
	}
	if anthropicTruthy(budget) {
		kwargs, err := n.anthropicBudgetEffort(budget)
		return kwargs, explicitBudget, err
	}
	if thinking.kind == Null && !anthropicTruthy(explicitBudget) && !thinkUnasked {
		return objectValue(Field{Name: "enable_thinking", Value: Value{kind: Boolean}}), explicitBudget, nil
	}
	return empty, explicitBudget, nil
}

func anthropicTruthy(v Value) bool {
	switch v.kind {
	case String:
		return v.text != ""
	case Boolean:
		return v.flag
	case Number:
		if !strings.ContainsAny(v.text, ".eE") {
			i, _ := new(big.Int).SetString(v.text, 10)
			return i.Sign() != 0
		}
		f, _ := strconv.ParseFloat(v.text, 64)
		return f != 0
	}
	return false
}

func (n *Normalizer) anthropicBudgetEffort(budget Value) (Value, error) {
	level := ""
	switch budget.kind {
	case Boolean:
		level = "low" // int(True) == 1; false does not enter this helper.
	case Number:
		if strings.ContainsAny(budget.text, ".eE") {
			num, _ := strconv.ParseFloat(budget.text, 64)
			if num < 2048 {
				level = "low"
			} else if num < 8192 {
				level = "medium"
			} else {
				level = "high"
			}
		} else {
			num, _ := new(big.Int).SetString(budget.text, 10)
			level = budgetIntegerLevel(num)
		}
	case String:
		for _, r := range budget.text {
			if r > 127 {
				return Value{}, fmt.Errorf("%w: non-ASCII thinking.budget_tokens", ErrUnsupported)
			}
		}
		s := strings.TrimSpace(budget.text)
		for i := 0; i < len(s); i++ {
			if s[i] == '_' && (i == 0 || i+1 == len(s) || s[i-1] < '0' || s[i-1] > '9' ||
				s[i+1] < '0' || s[i+1] > '9') {
				return objectValue(), nil
			}
		}
		s = strings.ReplaceAll(s, "_", "")
		if len(strings.TrimLeft(s, "+-")) > 4300 {
			return objectValue(), nil
		}
		num, ok := new(big.Int).SetString(s, 10)
		if !ok {
			return objectValue(), nil
		}
		level = budgetIntegerLevel(num)
	}
	if level == "" {
		return objectValue(), nil
	}
	if level == "high" && n.family != FamilyDeepSeek {
		level = "xhigh"
	}
	return objectValue(Field{Name: "reasoning_effort", Value: stringValue(level)}), nil
}

func budgetIntegerLevel(num *big.Int) string {
	if num.Cmp(big.NewInt(2048)) < 0 {
		return "low"
	}
	if num.Cmp(big.NewInt(8192)) < 0 {
		return "medium"
	}
	return "high"
}

// pinBillingStamp indexes Unicode code points, as Python's string operations do.
func pinBillingStamp(system string) string {
	const header = "x-anthropic-billing-header:"
	if !strings.HasPrefix(system, header) {
		return system
	}
	original := []rune(system)
	out := []rune(system)
	indexFrom := func(needle string, start int) int {
		at := strings.Index(string(original[start:]), needle)
		if at < 0 {
			return -1
		}
		return start + len([]rune(string(original[start:])[:at]))
	}
	if start := indexFrom("cch=", len([]rune(header))); start >= 0 && start <= 160 {
		value := start + len("cch=")
		if end := indexFrom(";", value); end > value && end-value <= 16 {
			for i := value; i < end; i++ {
				out[i] = 'f'
			}
		}
	}
	if start := indexFrom("cc_version=", 0); start >= 0 && start <= 160 {
		value := start + len("cc_version=")
		if end := indexFrom(";", value); end > value && end-value <= 64 {
			dots := 0
			for i := value; i < end; i++ {
				if original[i] == '.' {
					dots++
					if dots == 3 {
						for j := i + 1; j < end; j++ {
							out[j] = 'f'
						}
						break
					}
				}
			}
		}
	}
	return string(out)
}
