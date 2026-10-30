package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"strata/requestnorm"
)

const objectDirective = "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. " +
	"No Markdown, headings, code fences, commentary, or text outside the JSON. " +
	"Use every required field, correct types, and only allowed fields. " +
	"Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n"

func addObjectDirective(messages []requestnorm.Message, withTools bool, schema *outputSchema) []requestnorm.Message {
	text := `{"type":"object"}`
	if schema != nil {
		text = schema.text
	}
	directive := objectDirective + text
	if withTools {
		directive += "\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is " +
			"what you write once you are done with the tools."
	}
	if messages[0].Role == "system" {
		messages[0].Content.Text += "\n\n" + directive
		return messages
	}
	return append([]requestnorm.Message{{Role: "system", Content: requestnorm.TemplateContent{Text: directive}}}, messages...)
}

type structuredValidationError struct{ err error }

func (e *structuredValidationError) Error() string { return e.err.Error() }
func (e *structuredValidationError) Unwrap() error { return e.err }

// extractJSONObject mirrors the Python server's first-object extraction.
func extractJSONObject(text string) string {
	s := strings.TrimSpace(text)
	if strings.HasPrefix(s, "```") {
		if first := strings.IndexByte(s, '\n'); first >= 0 {
			if end := strings.Index(s[first+1:], "```"); end >= 0 {
				s = strings.TrimSpace(s[first+1 : first+1+end])
			}
		}
	}
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return s
	}
	depth, inString, escaped := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
		} else {
			switch c {
			case '"':
				inString = true
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return s[start : i+1]
				}
			}
		}
	}
	return s[start:]
}

func repairJSONEscapes(text string) string {
	var out strings.Builder
	for i := 0; i < len(text); i++ {
		if text[i] != '\\' {
			out.WriteByte(text[i])
			continue
		}
		if i+1 < len(text) && strings.ContainsRune(`\"/bfnrt`, rune(text[i+1])) {
			out.WriteString(text[i : i+2])
			i++
			continue
		}
		if i+5 < len(text) && text[i+1] == 'u' {
			if strings.Trim(text[i+2:i+6], "0123456789abcdefABCDEF") == "" {
				out.WriteString(text[i : i+6])
				i += 5
				continue
			}
		}
		out.WriteString(`\\`)
	}
	return out.String()
}

func canonicalJSON(value requestnorm.Value) (string, error) {
	switch value.Kind() {
	case requestnorm.Array:
		parts := make([]string, 0, len(value.Items()))
		for _, item := range value.Items() {
			part, err := canonicalJSON(item)
			if err != nil {
				return "", err
			}
			parts = append(parts, part)
		}
		return "[" + strings.Join(parts, ",") + "]", nil
	case requestnorm.Object:
		parts := make([]string, 0, len(value.Fields()))
		for _, field := range value.Fields() {
			quoted, err := json.Marshal(field.Name)
			if err != nil {
				return "", err
			}
			key, err := requestnorm.ParseValue(quoted)
			if err != nil {
				return "", err
			}
			name, err := key.MarshalJSON()
			if err != nil {
				return "", err
			}
			part, err := canonicalJSON(field.Value)
			if err != nil {
				return "", err
			}
			parts = append(parts, string(name)+":"+part)
		}
		return "{" + strings.Join(parts, ",") + "}", nil
	case requestnorm.Number:
		if n, ok := value.Float64(); ok {
			format := byte('f')
			if abs := math.Abs(n); abs != 0 && (abs < 1e-4 || abs >= 1e16) {
				format = 'e'
			}
			text := strconv.FormatFloat(n, format, -1, 64)
			if format == 'f' && !strings.Contains(text, ".") {
				text += ".0"
			}
			return text, nil
		}
	}
	b, err := value.MarshalJSON()
	return string(b), err
}

func validateJSONObject(text, finish string, repair bool) (string, error) {
	if finish != "stop" {
		return "", fmt.Errorf("structured output was incomplete (finish_reason=%s); increase the output budget", finish)
	}
	extracted := extractJSONObject(text)
	value, err := requestnorm.ParseValue([]byte(extracted))
	if err != nil && repair {
		value, err = requestnorm.ParseValue([]byte(repairJSONEscapes(extracted)))
	}
	if err != nil {
		return "", fmt.Errorf("model did not return valid JSON: %w", err)
	}
	if value.Kind() != requestnorm.Object {
		return "", errors.New("model output failed the JSON Schema at /: the answer is not a JSON object")
	}
	return canonicalJSON(value)
}
