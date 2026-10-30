package server

import (
	"errors"
	"strings"

	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
)

// normalizeForceChoice mirrors frontend.tool_choice_of and frontend.forced_call.
// The returned kind is auto, none, required, or named; prefix is the exact
// family-specific text that starts a forced call, or empty when no force applies.
func normalizeForceChoice(choice requestnorm.Value, tools []requestnorm.Value, family requestnorm.Family) (kind, prefix string) {
	kind = "auto"
	var name string
	switch choice.Kind() {
	case requestnorm.String:
		text, _ := choice.Text()
		switch text {
		case "auto", "none", "required":
			kind = text
		}
	case requestnorm.Object:
		typeValue, _ := choice.Lookup("type")
		if typeValue.Kind() == requestnorm.String {
			typ, _ := typeValue.Text()
			switch typ {
			case "auto", "none":
				kind = typ
			case "any", "required":
				kind = "required"
			case "function", "tool":
				kind = "named"
				fn, _ := choice.Lookup("function")
				if fn.Kind() == requestnorm.Object {
					value, _ := fn.Lookup("name")
					name, _ = nonemptyText(value)
				}
				if name == "" {
					value, _ := choice.Lookup("name")
					name, _ = nonemptyText(value)
				}
				if name == "" {
					kind = "auto"
				}
			}
		}
	}
	if kind == "auto" || kind == "none" {
		return kind, ""
	}

	names := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		value, ok := tool.Lookup("name")
		if !ok {
			continue
		}
		if text, ok := nonemptyText(value); ok {
			names[text] = struct{}{}
		}
	}
	if kind == "required" && len(names) == 1 {
		for name = range names {
		}
		kind = "named"
	}
	switch {
	case kind == "required" && len(names) > 0:
		prefix = "<tool_call>\n<function="
	case kind == "named":
		if _, ok := names[name]; ok {
			prefix = "<tool_call>\n<function=" + name + ">\n"
		} else {
			return "auto", ""
		}
	default:
		return "auto", ""
	}
	if family == requestnorm.FamilyDeepSeek {
		prefix = deepSeekForcePrefix(prefix)
	}
	return kind, prefix
}

func nonemptyText(value requestnorm.Value) (string, bool) {
	text, ok := value.Text()
	return text, ok && value.Kind() == requestnorm.String && text != ""
}

func deepSeekForcePrefix(force string) string {
	if force == "" {
		return force
	}
	head := "<｜DSML｜ calls>\n"
	const marker = "<function="
	at := strings.Index(force, marker)
	name := ""
	if at >= 0 {
		name = force[at+len(marker):]
	}
	if strings.HasSuffix(name, ">\n") {
		return head + "<｜DSML｜ invoke name=\"" + strings.TrimSuffix(name, ">\n") + "\">\n"
	}
	return head + "<｜DSML｜ invoke name=\"" + name
}

func appendForcedPrompt(tok *modeltokenizer.Tokenizer, rendered Rendered, ids []int, force string, maxBytes int) ([]int, error) {
	if force == "" {
		return ids, nil
	}
	if rendered.Thinking {
		return ids, nil
	}
	if len(rendered.Prompt) > maxBytes || len(force) > maxBytes-len(rendered.Prompt) {
		return nil, errors.New("prompt too large")
	}
	prefix, err := tok.EncodeWithBudget(force, true, nil, maxBytes-len(rendered.Prompt))
	if err != nil {
		return nil, err
	}
	return append(ids, prefix...), nil
}
