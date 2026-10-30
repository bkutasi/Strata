package outputparser

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

const dsml = "｜DSML｜"

var (
	dsmlCalls     = regexp.MustCompile(`^<｜DSML｜` + ws + `*(?:tool_|function_)?calls` + ws + `*>\z`)
	dsmlCallsEnd  = regexp.MustCompile(`^</｜DSML｜` + ws + `*(?:tool_|function_)?calls` + ws + `*>\z`)
	dsmlInvoke    = regexp.MustCompile(`^<｜DSML｜` + ws + `*invoke` + ws + `+name` + ws + `*=` + ws + `*"([^"]*)"` + ws + `*>\z`)
	dsmlInvokeEnd = regexp.MustCompile(`^</｜DSML｜` + ws + `*invoke` + ws + `*>\z`)
	dsmlParam     = regexp.MustCompile(`^<｜DSML｜` + ws + `*parameter` + ws + `+name` + ws + `*=` + ws + `*"([^"]*)"(?:` + ws + `+string` + ws + `*=` + ws + `*"(true|false)")?` + ws + `*>\z`)
	dsmlParamEnd  = regexp.MustCompile(`^</｜DSML｜` + ws + `*parameter` + ws + `*>\z`)
	dsmlPartial   = regexp.MustCompile(`^</?(?:｜(?:D(?:S(?:M(?:L(?:｜[^<>]*)?)?)?)?)?)?\z`)
)

// DSMLTranslator translates DeepSeek tool tags to the native parser's tags.
// Feed chunks must be valid UTF-8; partial tags are retained until complete.
type DSMLTranslator struct {
	pending     string
	state       string
	tool        string
	onParameter func(tool, name string, isString bool)
}

func NewDSMLTranslator(onParameter func(tool, name string, isString bool)) *DSMLTranslator {
	return &DSMLTranslator{state: "text", onParameter: onParameter}
}

func (t *DSMLTranslator) Feed(delta string) (string, error) {
	if !utf8.ValidString(delta) {
		return "", ErrInvalidUTF8
	}
	t.pending += delta
	return t.run(false), nil
}

func (t *DSMLTranslator) Flush() string { return t.run(true) }

func (t *DSMLTranslator) run(final bool) string {
	var out strings.Builder
	text := func(s string) {
		if (t.state == "calls" || t.state == "invoke") && strings.TrimFunc(s, space) == "" {
			return
		}
		out.WriteString(s)
	}
	for t.pending != "" {
		at := strings.IndexByte(t.pending, '<')
		if at < 0 {
			text(t.pending)
			t.pending = ""
			break
		}
		if at > 0 {
			text(t.pending[:at])
			t.pending = t.pending[at:]
			continue
		}
		end := strings.IndexByte(t.pending, '>')
		if end < 0 {
			if utf8.RuneCountInString(t.pending) <= 600 && dsmlPartial.MatchString(t.pending) {
				if final {
					text(t.pending)
					t.pending = ""
				}
				break
			}
		}
		if end >= 0 {
			head := t.pending[:end+1]
			if replacement, ok := t.tag(head); ok {
				out.WriteString(replacement)
				t.pending = t.pending[end+1:]
				continue
			}
		}
		text("<")
		t.pending = t.pending[1:]
	}
	return out.String()
}

func (t *DSMLTranslator) tag(head string) (string, bool) {
	if !strings.HasPrefix(head, "<"+dsml) && !strings.HasPrefix(head, "</"+dsml) {
		return "", false
	}
	switch {
	case dsmlParamEnd.MatchString(head):
		if t.state == "param" {
			t.state = "invoke"
			return "\n</parameter>\n", true
		}
		return "", false
	case dsmlInvokeEnd.MatchString(head), dsmlCallsEnd.MatchString(head):
		closingInvoke := dsmlInvokeEnd.MatchString(head)
		var out strings.Builder
		if t.state == "param" {
			out.WriteString("\n</parameter>\n")
			t.state = "invoke"
		}
		if closingInvoke && t.state == "invoke" {
			t.state = "calls"
			out.WriteString("</function>\n</tool_call>\n")
			return out.String(), true
		}
		if !closingInvoke && (t.state == "calls" || t.state == "invoke") {
			if t.state == "invoke" {
				out.WriteString("</function>\n</tool_call>\n")
			}
			t.state = "text"
			return out.String(), true
		}
		return "", false
	}
	if t.state == "param" {
		return "", false
	}
	if dsmlCalls.MatchString(head) {
		t.state = "calls"
		return "", true
	}
	if m := dsmlInvoke.FindStringSubmatch(head); m != nil {
		prefix := ""
		if t.state == "invoke" {
			prefix = "</function>\n</tool_call>\n"
		}
		t.state, t.tool = "invoke", m[1]
		return prefix + "<tool_call>\n<function=" + m[1] + ">\n", true
	}
	if m := dsmlParam.FindStringSubmatch(head); m != nil && (t.state == "calls" || t.state == "invoke") {
		t.state = "param"
		if t.onParameter != nil {
			t.onParameter(t.tool, m[1], m[2] != "false")
		}
		return "<parameter=" + m[1] + ">\n", true
	}
	return "", false
}

// DeepSeekParser combines DSML translation with the existing output parser.
type DeepSeekParser struct {
	parser     *Parser
	translator *DSMLTranslator
	reasoning  strings.Builder
	answered   bool
	fallback   bool
}

func NewDeepSeek(options Options) (*DeepSeekParser, error) {
	parser, err := New(options)
	if err != nil {
		return nil, err
	}
	p := &DeepSeekParser{parser: parser}
	p.translator = NewDSMLTranslator(p.onParameter)
	return p, nil
}

func (p *DeepSeekParser) onParameter(tool, name string, isString bool) {
	if !isString {
		return
	}
	schema, ok := p.parser.schemas[tool]
	if !ok {
		return
	}
	copyMap := func(m map[string]any) map[string]any {
		out := make(map[string]any, len(m)+1)
		for k, v := range m {
			out[k] = v
		}
		return out
	}
	spec := copyMap(schema)
	parameters, _ := spec["parameters"].(map[string]any)
	parameters = copyMap(parameters)
	properties, _ := parameters["properties"].(map[string]any)
	properties = copyMap(properties)
	property, _ := properties[name].(map[string]any)
	property = copyMap(property)
	property["type"] = "string"
	properties[name] = property
	parameters["properties"] = properties
	spec["parameters"] = parameters
	p.parser.schemas[tool] = spec
	if p.parser.scan.call != nil && p.parser.scan.call.Name == tool {
		p.parser.scan.declared[name] = "string"
	}
}

func (p *DeepSeekParser) watch(events []Event) []Event {
	for _, event := range events {
		switch event.Kind {
		case "reasoning":
			p.reasoning.WriteString(event.Text)
		case "tool_start", "tool_call":
			p.answered = true
		case "content":
			if strings.TrimFunc(event.Text, space) != "" {
				p.answered = true
			}
		}
	}
	return events
}

func (p *DeepSeekParser) Feed(delta string) ([]Event, error) {
	if err := p.parser.check(); err != nil {
		return nil, err
	}
	translated, err := p.translator.Feed(delta)
	if err != nil {
		return p.parser.fail(nil, err)
	}
	events, err := p.parser.Feed(translated)
	return p.watch(events), err
}

func (p *DeepSeekParser) Finish(reason string) ([]Event, error) {
	if err := p.parser.check(); err != nil {
		return nil, err
	}
	var out []Event
	if rest := p.translator.Flush(); rest != "" {
		events, err := p.parser.Feed(rest)
		out = append(out, p.watch(events)...)
		if err != nil {
			return out, err
		}
	}
	events, err := p.parser.Finish(reason)
	out = append(out, p.watch(events)...)
	if err != nil {
		return out, err
	}
	text := strings.TrimFunc(p.reasoning.String(), space)
	if (reason == "" || reason == "stop") && text != "" && !p.answered {
		p.answered, p.fallback = true, true
		out = append(out, Event{Kind: "content", Text: text})
	}
	return out, nil
}

func (p *DeepSeekParser) Status() Status {
	status := p.parser.Status()
	status.Clean = status.Clean && p.translator.pending == ""
	return status
}

// ReasoningFallback reports whether a natural stop used reasoning as the answer.
func (p *DeepSeekParser) ReasoningFallback() bool { return p.fallback }
