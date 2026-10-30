// Package responses implements a strict, model-free subset of Strata's
// stateless OpenAI Responses API conversion and output-event assembly.
package responses

import (
	"encoding/json"
	"fmt"
	"strings"

	"strata/requestnorm"
)

// Request is an immutable parsed Responses request. Decode rejects duplicate
// keys, lossy Unicode and unknown top-level fields.
type Request struct{ root requestnorm.Value }

// Decode parses a request. Call Prepare before rendering or serving it.
func Decode(data []byte) (*Request, error) {
	root, err := requestnorm.ParseValue(data)
	if err != nil {
		return nil, err
	}
	if err = known(root, "", "input", "instructions", "tools", "tool_choice", "reasoning", "text",
		"include", "model", "stream", "store", "temperature", "top_p", "parallel_tool_calls",
		"max_output_tokens", "metadata", "client_metadata", "prompt_cache_key", "stop", "stop_sequences",
		"previous_response_id", "conversation", "background", "prompt"); err != nil {
		return nil, err
	}
	return &Request{root: root}, nil
}

// ToolName maps a flattened template tool name back to a Responses tool.
type ToolName struct {
	Namespace string
	Name      string
	Kind      string // "function" or "custom"
}

// Prepared contains the prompt inputs and output-call mapping. Messages and
// Tools have the adjacent requestnorm module's template-compatible shapes.
type Prepared struct {
	Messages []requestnorm.Message
	Tools    []requestnorm.Value
	Names    map[string]ToolName
	Kwargs   requestnorm.Value
	Format   requestnorm.Value // null for plain text
}

// PrepareOptions selects the model family's effort vocabulary. A null
// SharedReasoningEffort means no server-wide default.
type PrepareOptions struct {
	Family                requestnorm.Family
	SharedReasoningEffort requestnorm.Value
}

// Prepare validates and converts a request without invoking a model. It does
// not enforce a non-empty conversation, as in the Python conversion layer.
func Prepare(req *Request, options PrepareOptions) (Prepared, error) {
	if req == nil {
		return Prepared{}, invalid("", "request is nil")
	}
	if err := checkRequest(req.root); err != nil {
		return Prepared{}, err
	}
	messages, err := inputMessages(req.root)
	if err != nil {
		return Prepared{}, err
	}
	tools, names, err := requestTools(req.root)
	if err != nil {
		return Prepared{}, err
	}
	format, err := textFormat(req.root)
	if err != nil {
		return Prepared{}, err
	}
	norm, err := requestnorm.NewNormalizer(options.Family)
	if err != nil {
		return Prepared{}, err
	}
	effort := field(field(req.root, "reasoning"), "effort")
	if effort.Kind() == requestnorm.Null {
		effort = options.SharedReasoningEffort
	}
	result, err := norm.NormalizeOpenAI(requestnorm.OpenAIRequest{ReasoningEffort: effort})
	if err != nil {
		return Prepared{}, invalid("reasoning.effort", err.Error())
	}
	return Prepared{Messages: messages, Tools: tools, Names: names, Kwargs: result.Kwargs, Format: format}, nil
}

func checkRequest(req requestnorm.Value) error {
	for _, name := range []string{"previous_response_id", "conversation", "background", "prompt"} {
		v := field(req, name)
		if v.Kind() != requestnorm.Null {
			if b, ok := v.Bool(); ok && !b {
				continue
			}
			if s, ok := stringOf(v); ok && s == "" {
				continue
			}
			return unsupported(name, "this stateless module does not support "+name)
		}
	}
	n := field(req, "max_output_tokens")
	if n.Kind() != requestnorm.Null {
		i, ok := n.Integer()
		if !ok || (strings.HasPrefix(i, "-") && i != "-1") {
			return invalid("max_output_tokens", "max_output_tokens must be a whole number, 1 or more, or -1 or 0 for no cap")
		}
	}
	reasoning := field(req, "reasoning")
	if reasoning.Kind() != requestnorm.Null {
		if err := known(reasoning, "reasoning", "effort", "summary"); err != nil {
			return err
		}
		if _, err := optionalString(field(reasoning, "summary"), "reasoning.summary"); err != nil {
			return err
		}
	}
	for _, name := range []string{"stream", "store", "parallel_tool_calls"} {
		v := field(req, name)
		if v.Kind() != requestnorm.Null {
			if _, ok := v.Bool(); !ok {
				return unsupported(name, name+" must be a boolean")
			}
		}
	}
	for _, name := range []string{"temperature", "top_p"} {
		v := field(req, name)
		if v.Kind() != requestnorm.Null && v.Kind() != requestnorm.Number {
			return unsupported(name, name+" must be a number or null")
		}
	}
	if _, err := optionalString(field(req, "model"), "model"); err != nil {
		return err
	}
	metadata := field(req, "metadata")
	if metadata.Kind() != requestnorm.Null && metadata.Kind() != requestnorm.Object {
		return unsupported("metadata", "metadata must be an object")
	}
	include := field(req, "include")
	if include.Kind() != requestnorm.Null {
		if include.Kind() != requestnorm.Array {
			return unsupported("include", "include must be an array")
		}
		for i, v := range include.Items() {
			s, ok := stringOf(v)
			if !ok || s != "reasoning.encrypted_content" {
				return unsupported(fmt.Sprintf("include[%d]", i), "unsupported include value")
			}
		}
	}
	return nil
}

// textFormat mirrors responses.text_format, but rejects extra settings rather
// than silently dropping them.
func textFormat(req requestnorm.Value) (requestnorm.Value, error) {
	text := field(req, "text")
	if text.Kind() == requestnorm.Null {
		return requestnorm.Value{}, nil
	}
	if err := known(text, "text", "format"); err != nil {
		return requestnorm.Value{}, err
	}
	format := field(text, "format")
	if format.Kind() == requestnorm.Null {
		return requestnorm.Value{}, nil
	}
	if err := known(format, "text.format", "type", "name", "schema", "strict", "description", "repair_latex_escapes"); err != nil {
		return requestnorm.Value{}, err
	}
	typ := field(format, "type")
	kind, ok := stringOf(typ)
	if typ.Kind() == requestnorm.Null || (ok && kind == "text") {
		if len(format.Fields()) > 1 {
			return requestnorm.Value{}, unsupported("text.format", "plain text format has unsupported settings")
		}
		return requestnorm.Value{}, nil
	}
	if !ok || (kind != "json_object" && kind != "json_schema") {
		return requestnorm.Value{}, invalid("text.format.type", "text.format.type must be text, json_object or json_schema")
	}
	repair := field(format, "repair_latex_escapes")
	if repair.Kind() != requestnorm.Null {
		if _, ok := repair.Bool(); !ok {
			return requestnorm.Value{}, unsupported("text.format.repair_latex_escapes", "repair_latex_escapes must be boolean")
		}
	}
	out := []requestnorm.Field{f("type", strValue(kind))}
	if kind == "json_schema" {
		var spec []requestnorm.Field
		for _, name := range []string{"name", "schema", "strict", "description"} {
			if v, present := format.Lookup(name); present {
				spec = append(spec, f(name, v))
			}
		}
		out = append(out, f("json_schema", object(spec...)))
	} else {
		for _, name := range []string{"name", "schema", "strict", "description"} {
			if _, present := format.Lookup(name); present {
				return requestnorm.Value{}, unsupported("text.format."+name, "unsupported json_object setting")
			}
		}
	}
	if b, ok := repair.Bool(); ok && b {
		out = append(out, f("repair_latex_escapes", repair))
	}
	return object(out...), nil
}

// parseArguments implements Python's incomplete-JSON fallback without
// discarding object member order.
func parseArguments(v requestnorm.Value, path string) (requestnorm.Value, error) {
	if v.Kind() == requestnorm.Object {
		return v, nil
	}
	if v.Kind() == requestnorm.Null {
		return object(), nil
	}
	s, ok := stringOf(v)
	if !ok {
		return requestnorm.Value{}, invalid(path, "function_call arguments must be a JSON string")
	}
	if strings.TrimSpace(s) == "" {
		return object(), nil
	}
	if !json.Valid([]byte(s)) {
		return object(f("arguments", strValue(s))), nil
	}
	value, err := requestnorm.ParseValue([]byte(s))
	if err != nil {
		return requestnorm.Value{}, fmt.Errorf("%s: %w", path, err)
	}
	if value.Kind() == requestnorm.Object {
		return value, nil
	}
	return object(f("arguments", value)), nil
}
