package responses

import (
	"encoding/base64"
	"fmt"
	"sort"
	"strings"

	"strata/requestnorm"
)

// EncryptedPrefix marks Strata's base64 reasoning replay payload. It is not
// encryption and must not be treated as confidential.
const EncryptedPrefix = "strata.r1:"

func content(v requestnorm.Value, path string) (requestnorm.TemplateContent, error) {
	if v.Kind() == requestnorm.Null {
		return requestnorm.TemplateContent{}, nil
	}
	if s, ok := stringOf(v); ok {
		return requestnorm.TemplateContent{Text: s}, nil
	}
	if v.Kind() != requestnorm.Array {
		return requestnorm.TemplateContent{}, invalid(path, "expected a string or an array of content parts")
	}
	var parts []requestnorm.Value
	var text strings.Builder
	hasImage := false
	for i, part := range v.Items() {
		p := fmt.Sprintf("%s[%d]", path, i)
		if s, ok := stringOf(part); ok {
			text.WriteString(s)
			parts = append(parts, object(f("type", strValue("text")), f("text", strValue(s))))
			continue
		}
		if part.Kind() != requestnorm.Object {
			return requestnorm.TemplateContent{}, invalid(p, "expected a content part object")
		}
		kind, _ := stringOf(field(part, "type"))
		switch kind {
		case "input_text", "output_text", "text", "summary_text", "reasoning_text", "refusal":
			if err := known(part, p, "type", "text", "refusal", "annotations", "logprobs"); err != nil {
				return requestnorm.TemplateContent{}, err
			}
			name := "text"
			if kind == "refusal" {
				name = "refusal"
			}
			other := "refusal"
			if kind == "refusal" {
				other = "text"
			}
			if _, present := part.Lookup(other); present {
				return requestnorm.TemplateContent{}, unsupported(p+"."+other, "unexpected content field")
			}
			for _, meta := range []string{"annotations", "logprobs"} {
				if x, present := part.Lookup(meta); present && x.Kind() != requestnorm.Array {
					return requestnorm.TemplateContent{}, unsupported(p+"."+meta, "content metadata must be an array")
				}
			}
			s, err := optionalString(field(part, name), p+"."+name)
			if err != nil {
				return requestnorm.TemplateContent{}, err
			}
			text.WriteString(s)
			parts = append(parts, object(f("type", strValue("text")), f("text", strValue(s))))
		case "input_image", "image_url":
			if err := known(part, p, "type", "image_url"); err != nil {
				return requestnorm.TemplateContent{}, err
			}
			source := field(part, "image_url")
			if source.Kind() == requestnorm.Object {
				if err := known(source, p+".image_url", "url"); err != nil {
					return requestnorm.TemplateContent{}, err
				}
				source = field(source, "url")
			}
			s, ok := stringOf(source)
			if !ok || s == "" {
				return requestnorm.TemplateContent{}, unsupported(p, "only images given as image_url are supported")
			}
			hasImage = true
			parts = append(parts, object(f("type", strValue("image")), f("source", strValue(s))))
		default:
			return requestnorm.TemplateContent{}, unsupported(p+".type", "unsupported content part type")
		}
	}
	if !hasImage {
		return requestnorm.TemplateContent{Text: text.String()}, nil
	}
	return requestnorm.TemplateContent{Parts: parts}, nil
}

func reasoningText(item requestnorm.Value, path string) (string, error) {
	c := field(item, "content")
	if c.Kind() != requestnorm.Null && c.Kind() != requestnorm.Array {
		return "", unsupported(path+".content", "reasoning content must be an array")
	}
	var texts []string
	for i, part := range c.Items() {
		p := fmt.Sprintf("%s.content[%d]", path, i)
		if err := known(part, p, "type", "text"); err != nil {
			return "", err
		}
		kind, _ := stringOf(field(part, "type"))
		if kind != "reasoning_text" && kind != "text" {
			return "", unsupported(p+".type", "unsupported reasoning part type")
		}
		s, err := optionalString(field(part, "text"), p+".text")
		if err != nil {
			return "", err
		}
		texts = append(texts, s)
	}
	if len(texts) > 0 {
		return strings.Join(texts, ""), nil
	}
	enc, err := optionalString(field(item, "encrypted_content"), path+".encrypted_content")
	if err != nil {
		return "", err
	}
	if !strings.HasPrefix(enc, EncryptedPrefix) {
		return "", nil // foreign payloads contain no replayable Strata text
	}
	data, err := base64.StdEncoding.Strict().DecodeString(strings.TrimPrefix(enc, EncryptedPrefix))
	if err == nil {
		var v requestnorm.Value
		v, err = requestnorm.ParseValue(data)
		if err == nil && v.Kind() == requestnorm.Object {
			if s, ok := stringOf(field(v, "text")); ok {
				return s, nil
			}
		}
	}
	return "", &Error{Message: "this reasoning item's encrypted_content is damaged",
		Param: path + ".encrypted_content", Code: "invalid_encrypted_content", Status: 400}
}

func inputMessages(req requestnorm.Value) ([]requestnorm.Message, error) {
	messages := make([]requestnorm.Message, 0)
	instructions := field(req, "instructions")
	if instructions.Kind() != requestnorm.Null {
		c, err := content(instructions, "instructions")
		if err != nil {
			return nil, err
		}
		if c.Parts != nil || c.Text != "" {
			messages = append(messages, requestnorm.Message{Role: "system", Content: c})
		}
	}
	input := field(req, "input")
	if input.Kind() == requestnorm.Null {
		return nil, &Error{Message: "input is required: a string or an array of input items",
			Param: "input", Code: "missing_required_parameter", Status: 400}
	}
	var items []requestnorm.Value
	if s, ok := stringOf(input); ok {
		items = []requestnorm.Value{object(f("type", strValue("message")), f("role", strValue("user")), f("content", strValue(s)))}
	} else if input.Kind() == requestnorm.Array {
		items = input.Items()
	} else {
		return nil, invalid("input", "input must be a string or an array of input items")
	}
	thinking := ""
	open := -1
	turn := func(c requestnorm.TemplateContent) {
		m := requestnorm.Message{Role: "assistant", Content: c, ReasoningContent: thinking}
		thinking = ""
		messages = append(messages, m)
		open = len(messages) - 1
	}
	for i, item := range items {
		p := fmt.Sprintf("input[%d]", i)
		if item.Kind() != requestnorm.Object {
			return nil, invalid(p, "expected an input item object")
		}
		kind, _ := stringOf(field(item, "type"))
		if kind == "" {
			if _, present := item.Lookup("role"); present {
				kind = "message"
			}
		}
		switch kind {
		case "message":
			if err := known(item, p, "type", "role", "content", "id", "status"); err != nil {
				return nil, err
			}
			role, _ := stringOf(field(item, "role"))
			if role != "user" && role != "assistant" && role != "system" && role != "developer" {
				return nil, invalid(p+".role", "unknown message role")
			}
			c, err := content(field(item, "content"), p+".content")
			if err != nil {
				return nil, err
			}
			if role == "assistant" {
				if c.Parts != nil {
					return nil, unsupported(p+".content", "assistant images would be discarded")
				}
				turn(c)
			} else {
				open, thinking = -1, ""
				if role == "developer" {
					role = "system"
				}
				messages = append(messages, requestnorm.Message{Role: role, Content: c})
			}
		case "reasoning":
			if err := known(item, p, "type", "id", "status", "content", "summary", "encrypted_content"); err != nil {
				return nil, err
			}
			summary := field(item, "summary")
			if summary.Kind() != requestnorm.Null && summary.Kind() != requestnorm.Array {
				return nil, unsupported(p+".summary", "reasoning summary must be an array")
			}
			text, err := reasoningText(item, p)
			if err != nil {
				return nil, err
			}
			if text != "" {
				if open >= 0 && (len(messages[open].ToolCalls) != 0 || messages[open].Content.Text != "") {
					open = -1
				}
				thinking += text
			}
		case "function_call", "custom_tool_call":
			if err := known(item, p, "type", "id", "status", "name", "namespace", "arguments", "input", "call_id"); err != nil {
				return nil, err
			}
			name, ok := stringOf(field(item, "name"))
			if !ok || name == "" {
				return nil, invalid(p+".name", "a function_call needs a name")
			}
			namespace, err := optionalString(field(item, "namespace"), p+".namespace")
			if err != nil {
				return nil, err
			}
			if namespace != "" {
				name = namespace + "." + name
			}
			var args requestnorm.Value
			if kind == "function_call" {
				if _, present := item.Lookup("input"); present {
					return nil, unsupported(p+".input", "input is only supported for custom calls")
				}
				args, err = parseArguments(field(item, "arguments"), p+".arguments")
			} else {
				if _, present := item.Lookup("arguments"); present {
					return nil, unsupported(p+".arguments", "arguments are only supported for function calls")
				}
				s, e := optionalString(field(item, "input"), p+".input")
				err = e
				args = object(f("input", strValue(s)))
			}
			if err != nil {
				return nil, err
			}
			id, err := optionalString(field(item, "call_id"), p+".call_id")
			if err != nil {
				return nil, err
			}
			if open < 0 || thinking != "" {
				turn(requestnorm.TemplateContent{})
			}
			messages[open].ToolCalls = append(messages[open].ToolCalls, requestnorm.ToolCall{
				ID: id, Function: requestnorm.FunctionCall{Name: name, Arguments: args},
			})
		case "function_call_output", "custom_tool_call_output":
			if err := known(item, p, "type", "id", "status", "call_id", "output"); err != nil {
				return nil, err
			}
			id, err := optionalString(field(item, "call_id"), p+".call_id")
			if err != nil {
				return nil, err
			}
			output := field(item, "output")
			var c requestnorm.TemplateContent
			if output.Kind() == requestnorm.Object {
				var s string
				s, err = pythonJSON(output)
				c = requestnorm.TemplateContent{Text: s}
			} else {
				c, err = content(output, p+".output")
			}
			if err != nil {
				return nil, err
			}
			messages = append(messages, requestnorm.Message{Role: "tool", Content: c, ToolCallID: id})
			open, thinking = -1, ""
		case "additional_tools":
			if err := known(item, p, "type", "role", "tools"); err != nil {
				return nil, err
			}
			if x := field(item, "tools"); x.Kind() != requestnorm.Null && x.Kind() != requestnorm.Array {
				return nil, unsupported(p+".tools", "additional tools must be an array")
			}
		case "item_reference":
			return nil, unsupported(p, "item references need stored responses")
		default:
			return nil, unsupported(p+".type", "unsupported input item type")
		}
	}
	orderResults(messages)
	lead := 0
	for lead < len(messages) && messages[lead].Role == "system" && messages[lead].Content.Parts == nil {
		lead++
	}
	if lead > 1 {
		texts := make([]string, lead)
		for i := range texts {
			texts[i] = messages[i].Content.Text
		}
		merged := requestnorm.Message{Role: "system", Content: requestnorm.TemplateContent{Text: strings.Join(texts, "\n\n")}}
		messages = append([]requestnorm.Message{merged}, messages[lead:]...)
	}
	for i := 1; i < len(messages); i++ {
		if messages[i].Role == "system" {
			messages[i].Role = "user"
		}
	}
	return messages, nil
}

func orderResults(messages []requestnorm.Message) {
	for i := 0; i < len(messages); {
		if messages[i].Role != "tool" {
			i++
			continue
		}
		j := i
		for j < len(messages) && messages[j].Role == "tool" {
			j++
		}
		rank := map[string]int{}
		count := 0
		if i > 0 {
			for n, c := range messages[i-1].ToolCalls {
				count++
				if c.ID != "" {
					rank[c.ID] = n
				}
			}
		}
		order := func(id string) int {
			if r, ok := rank[id]; ok {
				return r
			}
			return count
		}
		sort.SliceStable(messages[i:j], func(a, b int) bool {
			return order(messages[i+a].ToolCallID) < order(messages[i+b].ToolCallID)
		})
		i = j
	}
}
