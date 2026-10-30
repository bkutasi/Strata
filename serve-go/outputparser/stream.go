package outputparser

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"strata/toolparser"
)

type scanState struct {
	pos        int
	state      string
	call       *ToolCall
	first      bool
	valStarted bool
	declared   map[string]string
	search     needleSearch
	param      paramFinder
}

func (p *Parser) resetScan() {
	p.scan = scanState{state: "name", first: true, search: newNeedle(0)}
}

// jsonString implements the string part of Python's json.dumps, including
// ensure_ascii=True's surrogate-pair escapes.
func jsonString(s string, ascii bool) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 || ascii && r >= 0x7f {
				if r < 0x10000 {
					fmt.Fprintf(&b, `\u%04x`, r)
				} else {
					r -= 0x10000
					fmt.Fprintf(&b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
				}
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// Go maps have lost JSON object insertion order and Go float formatting is not
// Python's in all cases. Reject those shapes rather than emit different bytes.
func rawJSON(v any) (string, error) {
	switch x := v.(type) {
	case nil:
		return "null", nil
	case bool:
		return strconv.FormatBool(x), nil
	case string:
		return jsonString(x, false), nil
	case json.Number:
		return string(x), nil
	case []any:
		values := make([]string, 0, len(x))
		for _, item := range x {
			s, err := rawJSON(item)
			if err != nil {
				return "", err
			}
			values = append(values, s)
		}
		return "[" + strings.Join(values, ", ") + "]", nil
	case map[string]any:
		if len(x) > 1 {
			return "", fmt.Errorf("%w: streamed JSON object with multiple keys", ErrUnsupported)
		}
		for key, value := range x {
			s, err := rawJSON(value)
			if err != nil {
				return "", err
			}
			return "{" + jsonString(key, false) + ": " + s + "}", nil
		}
		return "{}", nil
	case float64:
		return "", fmt.Errorf("%w: streamed floating-point value", ErrUnsupported)
	default:
		return "", fmt.Errorf("%w: streamed value %T", ErrUnsupported, v)
	}
}

func (p *Parser) scanCall(data []byte) ([]Event, error) {
	var out []Event
	var work *int
	if p.work != nil {
		work = &p.work.argument
	}
	args := func(text string) {
		if text != "" {
			out = append(out, Event{Kind: "tool_args", Text: text, Call: p.scan.call})
		}
	}
	for {
		rest := data[p.scan.pos:]
		switch p.scan.state {
		case "name":
			a := p.scan.search.find(data, toolparser.FuncStart, work)
			if a < 0 {
				return out, nil
			}
			p.scan.search = newNeedle(a + len(toolparser.FuncStart))
			p.scan.pos = a + len(toolparser.FuncStart)
			p.scan.state = "name_end"
		case "name_end":
			b := p.scan.search.find(data, ">", work)
			if b < 0 {
				return out, nil
			}
			raw := string(data[p.scan.pos:b])
			name := p.canonical(strip(raw))
			if raw != strip(raw) {
				if _, ok := p.schemas[name]; !ok {
					return out, nil
				}
			}
			call, err := p.newCall(name, map[string]any{})
			if err != nil {
				return out, err
			}
			p.scan.call = call
			p.scan.declared = map[string]string{}
			params, _ := p.schemas[name]["parameters"].(map[string]any)
			props, _ := params["properties"].(map[string]any)
			for k, v := range props {
				property, _ := v.(map[string]any)
				typ, _ := property["type"].(string)
				p.scan.declared[k] = typ
			}
			out = append(out, Event{Kind: "tool_start", Call: call})
			args("{")
			p.scan.pos = b + 1
			p.scan.state = "between"
		case "between":
			p.scan.pos = skipSpace(data, p.scan.pos, work)
			rest = data[p.scan.pos:]
			if prefix(rest, toolparser.ParamStart, work) == 1 {
				p.scan.pos += len(toolparser.ParamStart)
				p.scan.search = newNeedle(p.scan.pos)
				p.scan.state = "param_name"
			} else if prefix(rest, toolparser.FuncEnd, work) == 1 {
				p.scan.pos += len(toolparser.FuncEnd)
				p.scan.state = "function_done"
			} else {
				return out, nil
			}
		case "param_name":
			b := p.scan.search.find(data, ">", work)
			if b < 0 {
				return out, nil
			}
			name := string(data[p.scan.pos:b])
			prefix := ""
			if !p.scan.first {
				prefix = ","
			}
			args(prefix + jsonString(name, true) + ":")
			p.scan.first = false
			p.scan.pos = b + 1
			p.scan.param = newParamFinder(p.scan.pos)
			if p.scan.declared[name] == "string" {
				args(`"`)
				p.scan.state, p.scan.valStarted = "str", false
			} else {
				p.scan.state = "raw"
			}
		case "str":
			if !p.scan.valStarted {
				if len(rest) == 0 {
					return out, nil
				}
				if rest[0] == '\n' {
					p.scan.pos++
					rest = rest[1:]
					p.scan.param = newParamFinder(p.scan.pos)
				}
				p.scan.valStarted = true
			}
			end := p.scan.param.find(data, work)
			if end >= 0 {
				value := strings.TrimSuffix(string(data[p.scan.pos:end]), "\n")
				quoted := jsonString(value, true)
				args(quoted[1:len(quoted)-1] + `"`)
				p.scan.pos = end + len(toolparser.ParamEnd)
				p.scan.state = "between"
				continue
			}
			safe := len(rest) - hold(string(rest[max(0, len(rest)-len(toolparser.ParamEnd)+1):]), toolparser.ParamEnd)
			if end == -2 {
				safe = p.scan.param.first - p.scan.pos
			}
			if safe > 0 && rest[safe-1] == '\n' {
				safe--
			}
			if safe > 0 {
				if !utf8.Valid(rest[:safe]) {
					return out, ErrInvalidUTF8
				}
				quoted := jsonString(string(rest[:safe]), true)
				args(quoted[1 : len(quoted)-1])
				p.scan.pos += safe
				if p.scan.param.first < p.scan.pos {
					p.scan.param.first = -1
				}
			}
			return out, nil
		case "raw":
			end := p.scan.param.find(data, work)
			if end < 0 {
				return out, nil
			}
			value := string(data[p.scan.pos:end])
			body := "<function=f><parameter=x>" + value + "</parameter></function>"
			call, err := toolparser.ParseToolCall(body, nil)
			if err != nil {
				return out, err
			}
			encoded, err := rawJSON(call.Arguments["x"])
			if err != nil {
				return out, err
			}
			args(encoded)
			p.scan.pos = end + len(toolparser.ParamEnd)
			p.scan.state = "between"
		default:
			return out, nil
		}
	}
}

func (p *Parser) closeScan(data []byte) []Event {
	if p.scan.call == nil || p.scan.state == "done" {
		return nil
	}
	tail := ""
	if p.scan.state == "str" {
		tail += `"`
	} else if p.scan.state == "raw" {
		tail += jsonString(strings.Trim(string(data[p.scan.pos:]), "\n"), true)
	}
	tail += "}"
	p.scan.state = "done"
	return []Event{{Kind: "tool_args", Text: tail, Call: p.scan.call}}
}
