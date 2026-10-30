package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"unicode"

	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
)

var literalToolTags = map[string]bool{
	"<tool_call>": true, "</tool_call>": true,
	"<tool_response>": true, "</tool_response>": true,
}

func literalTags(controls []string) []string {
	tags := []string{"<think>", "</think>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>",
		"<|video_pad|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"}
	seen := make(map[string]bool, len(tags)+len(controls))
	for _, tag := range tags {
		seen[tag] = true
	}
	for _, tag := range controls {
		if tag != "" && !seen[tag] {
			tags = append(tags, tag)
			seen[tag] = true
		}
	}
	sort.SliceStable(tags, func(i, j int) bool { return len(tags[i]) > len(tags[j]) })
	return tags
}

type literalMarker struct {
	inputs    [3]requestnorm.Value
	all, own  *strings.Replacer
	restoreTo map[rune]string
}

func newLiteralMarker(tags []string, source string, inputs ...[]byte) (*literalMarker, error) {
	m := &literalMarker{}
	found := false
	invalidKey := false
	used := make(map[rune]bool)
	for _, ch := range source {
		if ch >= 0xf0000 {
			used[ch] = true
		}
	}
	var contains func(requestnorm.Value)
	contains = func(v requestnorm.Value) {
		switch v.Kind() {
		case requestnorm.String:
			text, _ := v.Text()
			for _, ch := range text {
				if ch >= 0xf0000 {
					used[ch] = true
				}
			}
			for _, tag := range tags {
				if strings.Contains(text, tag) {
					found = true
				}
			}
		case requestnorm.Array:
			for _, item := range v.Items() {
				contains(item)
			}
		case requestnorm.Object:
			for _, field := range v.Fields() {
				for _, ch := range field.Name {
					if ch >= 0xf0000 {
						used[ch] = true
					}
				}
				for _, tag := range tags {
					if strings.Contains(field.Name, tag) {
						invalidKey = true
					}
				}
				contains(field.Value)
			}
		}
	}
	for i, input := range inputs {
		v, err := requestnorm.ParseValue(input)
		if err != nil {
			return nil, err
		}
		m.inputs[i] = v
		contains(v)
	}
	if invalidKey {
		return nil, errors.New("control-token literals in object keys are not supported")
	}
	if !found {
		return nil, nil
	}
	var all, own []string
	m.restoreTo = make(map[rune]string, len(tags))
	ch := rune(0xf0000)
	for _, tag := range tags {
		for ch <= 0x10fffd && (used[ch] || ch == 0xffffe || ch == 0xfffff) {
			ch++
		}
		if ch > 0x10fffd {
			return nil, errors.New("no available literal markers")
		}
		m.restoreTo[ch] = tag
		all = append(all, tag, string(ch))
		if !literalToolTags[tag] {
			own = append(own, tag, string(ch))
		}
		ch++
	}
	m.all, m.own = strings.NewReplacer(all...), strings.NewReplacer(own...)
	return m, nil
}

func (m *literalMarker) text(text string, assistant bool) string {
	if assistant {
		return m.own.Replace(text)
	}
	return m.all.Replace(text)
}

func (m *literalMarker) contentText(text, role string) string {
	if role == "assistant" && strings.HasPrefix(strings.TrimLeftFunc(text, unicode.IsSpace), "<think>") {
		start := strings.Index(text, "<think>") + len("<think>")
		end := strings.Index(text, "</think>")
		if end >= start {
			return text[:start] + m.text(text[start:end], true) + "</think>" + m.text(text[end+len("</think>"):], true)
		}
	}
	if role == "user" {
		const open, close = "<tool_response>", "</tool_response>"
		trimmed := strings.TrimSpace(text)
		if len(trimmed) >= len(open)+len(close) && strings.HasPrefix(trimmed, open) && strings.HasSuffix(trimmed, close) {
			start, end := strings.Index(text, open), strings.LastIndex(text, close)
			return m.text(text[:start], false) + open + m.text(text[start+len(open):end], false) + close + m.text(text[end+len(close):], false)
		}
	}
	return m.text(text, role == "assistant")
}

func writeJSONText(b *bytes.Buffer, text string) {
	encoded, _ := json.Marshal(text)
	b.Write(encoded)
}

func (m *literalMarker) value(b *bytes.Buffer, v requestnorm.Value, assistant bool) error {
	switch v.Kind() {
	case requestnorm.String:
		text, _ := v.Text()
		writeJSONText(b, m.text(text, assistant))
	case requestnorm.Array:
		b.WriteByte('[')
		for i, item := range v.Items() {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := m.value(b, item, assistant); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case requestnorm.Object:
		b.WriteByte('{')
		for i, field := range v.Fields() {
			if i > 0 {
				b.WriteByte(',')
			}
			writeJSONText(b, field.Name)
			b.WriteByte(':')
			if err := m.value(b, field.Value, assistant); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		data, err := v.MarshalJSON()
		if err != nil {
			return err
		}
		b.Write(data)
	}
	return nil
}

func (m *literalMarker) content(b *bytes.Buffer, v requestnorm.Value, role string) error {
	if v.Kind() == requestnorm.String {
		text, _ := v.Text()
		writeJSONText(b, m.contentText(text, role))
		return nil
	}
	if role == "user" && v.Kind() == requestnorm.Array {
		items := v.Items()
		texts := make([]string, len(items))
		valid := len(items) > 0
		for i, item := range items {
			text, ok := item.Lookup("text")
			texts[i], ok = text.Text()
			if item.Kind() != requestnorm.Object || !ok || text.Kind() != requestnorm.String {
				valid = false
				break
			}
		}
		const open, close = "<tool_response>", "</tool_response>"
		joined := strings.TrimSpace(strings.Join(texts, ""))
		valid = valid && len(joined) >= len(open)+len(close) &&
			strings.HasPrefix(joined, open) && strings.HasSuffix(joined, close) &&
			strings.HasPrefix(strings.TrimLeftFunc(texts[0], unicode.IsSpace), open) &&
			strings.HasSuffix(strings.TrimRightFunc(texts[len(texts)-1], unicode.IsSpace), close)
		if valid {
			b.WriteByte('[')
			for i, item := range items {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteByte('{')
				for j, field := range item.Fields() {
					if j > 0 {
						b.WriteByte(',')
					}
					writeJSONText(b, field.Name)
					b.WriteByte(':')
					if field.Name != "text" {
						if err := m.value(b, field.Value, false); err != nil {
							return err
						}
						continue
					}
					text := texts[i]
					switch {
					case len(items) == 1:
						text = m.contentText(text, role)
					case i == 0:
						start := strings.Index(text, open)
						text = m.text(text[:start], false) + open + m.text(text[start+len(open):], false)
					case i == len(items)-1:
						end := strings.LastIndex(text, close)
						text = m.text(text[:end], false) + close + m.text(text[end+len(close):], false)
					default:
						text = m.text(text, false)
					}
					writeJSONText(b, text)
				}
				b.WriteByte('}')
			}
			b.WriteByte(']')
			return nil
		}
	}
	return m.value(b, v, role == "assistant")
}

func (m *literalMarker) messages(b *bytes.Buffer, v requestnorm.Value) error {
	b.WriteByte('[')
	for i, message := range v.Items() {
		if i > 0 {
			b.WriteByte(',')
		}
		roleValue, _ := message.Lookup("role")
		role, _ := roleValue.Text()
		b.WriteByte('{')
		for j, field := range message.Fields() {
			if j > 0 {
				b.WriteByte(',')
			}
			writeJSONText(b, field.Name)
			b.WriteByte(':')
			if field.Name == "role" {
				if err := m.value(b, field.Value, false); err != nil {
					return err
				}
			} else if field.Name == "content" {
				if err := m.content(b, field.Value, role); err != nil {
					return err
				}
			} else if err := m.value(b, field.Value, role == "assistant"); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	}
	b.WriteByte(']')
	return nil
}

func (m *literalMarker) markInputs() ([]byte, []byte, []byte, error) {
	var messages, tools, kwargs bytes.Buffer
	if err := m.messages(&messages, m.inputs[0]); err != nil {
		return nil, nil, nil, err
	}
	if err := m.value(&tools, m.inputs[1], false); err != nil {
		return nil, nil, nil, err
	}
	if err := m.value(&kwargs, m.inputs[2], false); err != nil {
		return nil, nil, nil, err
	}
	return messages.Bytes(), tools.Bytes(), kwargs.Bytes(), nil
}

func (m *literalMarker) restore(prompt string) (string, []modeltokenizer.Span) {
	var out strings.Builder
	var spans []modeltokenizer.Span
	pos := 0
	for i, ch := range prompt {
		if tag, ok := m.restoreTo[ch]; ok {
			out.WriteString(prompt[pos:i])
			start := out.Len()
			out.WriteString(tag)
			spans = append(spans, modeltokenizer.Span{Start: start, End: out.Len()})
			pos = i + len(string(ch))
		}
	}
	out.WriteString(prompt[pos:])
	return out.String(), spans
}
