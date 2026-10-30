package responses

import (
	"fmt"

	"strata/requestnorm"
)

func requestTools(req requestnorm.Value) ([]requestnorm.Value, map[string]ToolName, error) {
	given := field(req, "tools")
	if given.Kind() != requestnorm.Null && given.Kind() != requestnorm.Array {
		return nil, nil, invalid("tools", "tools must be an array")
	}
	all := given.Items()
	input := field(req, "input")
	for _, item := range input.Items() {
		kind, _ := stringOf(field(item, "type"))
		if kind == "additional_tools" {
			all = append(all, field(item, "tools").Items()...)
		}
	}
	var tools []requestnorm.Value
	names := map[string]ToolName{}
	add := func(tool requestnorm.Value, path, namespace, nsDescription string) error {
		if err := known(tool, path, "type", "name", "description", "parameters", "strict", "format"); err != nil {
			return err
		}
		kind, _ := stringOf(field(tool, "type"))
		if kind == "" {
			kind = "function"
		}
		name, ok := stringOf(field(tool, "name"))
		if !ok || name == "" {
			return invalid(path+".name", "a tool needs a name")
		}
		description, err := optionalString(field(tool, "description"), path+".description")
		if err != nil {
			return err
		}
		if nsDescription != "" {
			if description == "" {
				description = nsDescription
			} else {
				description = nsDescription + "\n\n" + description
			}
		}
		flat := name
		if namespace != "" {
			flat = namespace + "." + name
		}
		if _, exists := names[flat]; exists {
			return unsupported(path+".name", "duplicate flattened tool name")
		}
		var params requestnorm.Value
		switch kind {
		case "function":
			if _, present := tool.Lookup("format"); present {
				return unsupported(path+".format", "format is only supported for custom tools")
			}
			params = field(tool, "parameters")
			if params.Kind() == requestnorm.Null || (params.Kind() == requestnorm.Object && len(params.Fields()) == 0) {
				params = parse(`{"type":"object","properties":{}}`)
			} else if params.Kind() != requestnorm.Object {
				return unsupported(path+".parameters", "function parameters must be an object")
			}
		case "custom":
			if _, present := tool.Lookup("parameters"); present {
				return unsupported(path+".parameters", "parameters are only supported for function tools")
			}
			format := field(tool, "format")
			if format.Kind() != requestnorm.Null {
				if err := known(format, path+".format", "type", "syntax", "definition"); err != nil {
					return err
				}
				typ, _ := stringOf(field(format, "type"))
				if typ != "grammar" && typ != "text" {
					return unsupported(path+".format.type", "unsupported custom tool format")
				}
				syntax, err := optionalString(field(format, "syntax"), path+".format.syntax")
				if err != nil {
					return err
				}
				definition, err := optionalString(field(format, "definition"), path+".format.definition")
				if err != nil {
					return err
				}
				if typ == "grammar" && definition != "" {
					description += "\n\nThe input must follow this " + syntax + " grammar:\n" + definition
				}
			}
			params = parse(`{"type":"object","properties":{"input":{"type":"string","description":"the raw input"}},"required":["input"]}`)
		default:
			return unsupported(path+".type", "unsupported tool type")
		}
		tools = append(tools, object(f("name", strValue(flat)), f("description", strValue(description)), f("parameters", params)))
		names[flat] = ToolName{Namespace: namespace, Name: name, Kind: kind}
		return nil
	}
	for i, tool := range all {
		p := fmt.Sprintf("tools[%d]", i)
		if tool.Kind() != requestnorm.Object {
			return nil, nil, invalid(p, "expected a tool object")
		}
		kind, ok := stringOf(field(tool, "type"))
		if !ok {
			return nil, nil, invalid(p+".type", "a tool needs a type")
		}
		switch kind {
		case "function", "custom":
			if err := add(tool, p, "", ""); err != nil {
				return nil, nil, err
			}
		case "namespace":
			if err := known(tool, p, "type", "name", "description", "tools"); err != nil {
				return nil, nil, err
			}
			ns, ok := stringOf(field(tool, "name"))
			if !ok || ns == "" {
				return nil, nil, unsupported(p+".name", "namespace needs a name")
			}
			desc, err := optionalString(field(tool, "description"), p+".description")
			if err != nil {
				return nil, nil, err
			}
			members := field(tool, "tools")
			if members.Kind() != requestnorm.Null && members.Kind() != requestnorm.Array {
				return nil, nil, unsupported(p+".tools", "namespace tools must be an array")
			}
			for j, member := range members.Items() {
				path := fmt.Sprintf("%s.tools[%d]", p, j)
				kind := field(member, "type")
				if kind.Kind() != requestnorm.Null {
					s, _ := stringOf(kind)
					if s != "function" && s != "custom" {
						return nil, nil, invalid(path, "a namespace holds function tools")
					}
				}
				if err := add(member, path, ns, desc); err != nil {
					return nil, nil, err
				}
			}
		default:
			// Python skips hosted and unknown tools. Refusing them avoids silently
			// changing the tools a client believes it offered.
			return nil, nil, unsupported(p+".type", "hosted or unknown tools are not supported")
		}
	}
	if s, _ := stringOf(field(req, "tool_choice")); s == "none" {
		return nil, names, nil
	}
	return tools, names, nil
}
