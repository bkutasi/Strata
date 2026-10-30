package server

import (
	"errors"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"strata/requestnorm"
)

type outputSchema struct {
	text     string
	compiled *jsonschema.Schema
}

func schemaOnlyObjects(node, root requestnorm.Value, seen map[string]bool) bool {
	if node.Kind() != requestnorm.Object {
		return false
	}
	typ, _ := node.Lookup("type")
	if text, ok := typ.Text(); ok && typ.Kind() == requestnorm.String && text == "object" {
		return true
	}
	ref, _ := node.Lookup("$ref")
	if text, ok := ref.Text(); ok && ref.Kind() == requestnorm.String && strings.HasPrefix(text, "#") && !seen[text] {
		target := root
		for _, part := range strings.Split(text[1:], "/")[1:] {
			part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
			var found bool
			target, found = target.Lookup(part)
			if !found {
				return false
			}
		}
		seen[text] = true
		result := schemaOnlyObjects(target, root, seen)
		delete(seen, text)
		return result
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		branches, _ := node.Lookup(key)
		if branches.Kind() == requestnorm.Array && len(branches.Items()) > 0 {
			all := true
			for _, branch := range branches.Items() {
				if !schemaOnlyObjects(branch, root, seen) {
					all = false
					break
				}
			}
			if all {
				return true
			}
		}
	}
	branches, _ := node.Lookup("allOf")
	for _, branch := range branches.Items() {
		if schemaOnlyObjects(branch, root, seen) {
			return true
		}
	}
	return false
}

func checkLocalSchemaReferences(value requestnorm.Value) error {
	switch value.Kind() {
	case requestnorm.Object:
		for _, field := range value.Fields() {
			if field.Name == "$ref" || field.Name == "$dynamicRef" {
				if text, ok := field.Value.Text(); ok && field.Value.Kind() == requestnorm.String && !strings.HasPrefix(text, "#") {
					return errors.New("response_format supports only local schema references")
				}
			}
			if err := checkLocalSchemaReferences(field.Value); err != nil {
				return err
			}
		}
	case requestnorm.Array:
		for _, item := range value.Items() {
			if err := checkLocalSchemaReferences(item); err != nil {
				return err
			}
		}
	}
	return nil
}

func compileOutputSchema(format requestnorm.Value) (*outputSchema, error) {
	spec, _ := format.Lookup("json_schema")
	schema, _ := spec.Lookup("schema")
	if spec.Kind() != requestnorm.Object || schema.Kind() != requestnorm.Object {
		return nil, errors.New("response_format.json_schema needs a schema object")
	}
	name, _ := spec.Lookup("name")
	text, ok := name.Text()
	if !ok || name.Kind() != requestnorm.String || text == "" {
		return nil, errors.New("response_format.json_schema needs a name")
	}
	if strict, present := spec.Lookup("strict"); present {
		if _, ok := strict.Bool(); !ok {
			return nil, errors.New("response_format.json_schema.strict must be boolean")
		}
	}
	if !schemaOnlyObjects(schema, schema, make(map[string]bool)) {
		return nil, errors.New("response_format schema must accept only JSON objects at its root")
	}
	if err := checkLocalSchemaReferences(schema); err != nil {
		return nil, err
	}
	serialized, err := canonicalJSON(schema)
	if err != nil {
		return nil, err
	}
	document, err := jsonschema.UnmarshalJSON(strings.NewReader(serialized))
	if err != nil {
		return nil, err
	}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	// Never allow schemas to read local files or fetch external resources.
	compiler.UseLoader(jsonschema.SchemeURLLoader{})
	const location = "https://strata.invalid/schema"
	if err = compiler.AddResource(location, document); err != nil {
		return nil, err
	}
	compiled, err := compiler.Compile(location)
	if err != nil {
		return nil, fmt.Errorf("invalid response_format schema: %w", err)
	}
	return &outputSchema{text: serialized, compiled: compiled}, nil
}

func validateStructuredJSON(text, finish string, repair bool, schema *outputSchema) (string, error) {
	canonical, err := validateJSONObject(text, finish, repair)
	if err != nil || schema == nil {
		return canonical, err
	}
	value, err := jsonschema.UnmarshalJSON(strings.NewReader(canonical))
	if err != nil {
		return "", err
	}
	if err = schema.compiled.Validate(value); err != nil {
		return "", fmt.Errorf("model output failed the JSON Schema: %w", err)
	}
	return canonical, nil
}
