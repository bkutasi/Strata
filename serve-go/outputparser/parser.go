// Package outputparser incrementally parses the Qwen-style Strata output
// protocol. See README.md for supported modes and compatibility limits.
package outputparser

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"strata/toolparser"
)

const (
	thinkEnd  = "</think>"
	callStart = "<tool_call>"
	callEnd   = "</tool_call>"
	rcallMax  = 32768
)

var (
	// ErrUnsupported marks a deliberately unimplemented compatibility branch.
	ErrUnsupported = errors.New("output parser: unsupported behavior")
	// ErrClosed is returned after Finish or a terminal parser error.
	ErrClosed = errors.New("output parser: parser is closed")
	// ErrInvalidUTF8 indicates a chunk or configuration that cannot be represented
	// as a Python Unicode string without loss.
	ErrInvalidUTF8 = errors.New("output parser: invalid UTF-8")

	ws           = `[\t-\r\x{001c}-\x{001f} \p{Z}\x{0085}]`
	nonWS        = `[^<>\t-\r\x{001c}-\x{001f} \p{Z}\x{0085}]`
	functionOpen = regexp.MustCompile(`^<function=(` + ws + `*` + nonWS + `+` + ws + `*)>`)
	functionPart = regexp.MustCompile(`^<function=` + ws + `*` + nonWS + `*` + ws + `*$`)
	looseOpen    = regexp.MustCompile(`^<` + ws + `*(?:function` + ws + `*[=:]` + ws + `*|=` + ws + `*)?([A-Za-z0-9_.\-]+)` + ws + `*>`)
	loosePart    = regexp.MustCompile(`^<` + ws + `*(?:function[ \t]*[=:]?` + ws + `*|=` + ws + `*)?[A-Za-z0-9_.\-]*` + ws + `*$`)
)

// ToolCall is a final parsed call or the identity attached to streamed pieces.
// Arguments in tool_start/tool_args are empty, as in Python's streaming view.
type ToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
	ID        string         `json:"id"`
}

// Event is ordered within and across Feed/Finish return values. tool_start and
// tool_args are provisional and must not be executed; wait for tool_call.
// Events returned with an error must not be used to execute a call.
type Event struct {
	Kind string    `json:"kind"`
	Text string    `json:"text,omitempty"`
	Call *ToolCall `json:"call,omitempty"`
}

// Options configures one parser. Tools use the Python frontend's normalized
// schema shape, e.g. {"name":"f","parameters":{"properties":{...}}}.
// Treat Tools and nested values as immutable during the parser's lifetime.
type Options struct {
	Thinking     bool
	Tools        []map[string]any
	StreamTools  bool
	Recover      bool
	DeclaredOnly bool
	// ID, when provided, generates each new call ID. Supplying a deterministic
	// generator allows identical inputs and chunk boundaries to be replayed.
	ID func() (string, error)
}

type pendingCall struct {
	raw        string
	whitespace []byte
	call       *ToolCall
}

// Parser is a single-use state machine; it is not safe for concurrent use.
type Parser struct {
	state, buf  string
	lead        bool
	schemas     map[string]map[string]any
	aliases     map[string]string
	declared    bool
	stream      bool
	id          func() (string, error)
	pending     []pendingCall
	fence       byte
	line        []byte
	ticks       int
	unread      int
	unreadNames []string
	rescued     int
	refused     int
	closed      bool
	failure     error
	scan        scanState
	callBuf     []byte
	callRunes   int
	frame       frameScanner
	thinkSearch needleSearch
	opener      *openerHold
	work        *workCounts
}

// New constructs a parser. Recovery is rejected rather than silently behaving
// like the default mode.
func New(options Options) (*Parser, error) {
	if options.Recover {
		return nil, fmt.Errorf("%w: recover=true", ErrUnsupported)
	}
	p := &Parser{
		state: "content", schemas: make(map[string]map[string]any),
		aliases: make(map[string]string), declared: options.DeclaredOnly,
		stream: options.StreamTools, id: options.ID,
	}
	if options.Thinking {
		p.state = "reasoning"
	}
	if p.id == nil {
		p.id = func() (string, error) {
			var b [12]byte
			if _, err := rand.Read(b[:]); err != nil {
				return "", err
			}
			b[6] = b[6]&0x0f | 0x40
			b[8] = b[8]&0x3f | 0x80
			return "call_" + hex.EncodeToString(b[:]), nil
		}
	}
	for _, schema := range options.Tools {
		name, ok := schema["name"].(string)
		if !ok {
			return nil, errors.New("output parser: each tool must have a string name")
		}
		if !utf8.ValidString(name) {
			return nil, ErrInvalidUTF8
		}
		if value := schema["parameters"]; value != nil {
			params, ok := value.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("output parser: tool %q parameters must be an object", name)
			}
			if value := params["properties"]; value != nil {
				props, ok := value.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("output parser: tool %q properties must be an object", name)
				}
				for key, value := range props {
					if !utf8.ValidString(key) {
						return nil, ErrInvalidUTF8
					}
					if value != nil {
						if _, ok := value.(map[string]any); !ok {
							return nil, fmt.Errorf("output parser: tool %q property %q must be an object", name, key)
						}
					}
				}
			}
		}
		p.schemas[name] = schema
	}
	candidates := make(map[string]map[string]bool)
	add := func(alias, name string) {
		if candidates[alias] == nil {
			candidates[alias] = make(map[string]bool)
		}
		candidates[alias][name] = true
	}
	for name := range p.schemas {
		if dot := strings.LastIndexByte(name, '.'); dot >= 0 {
			ns, member := name[:dot], name[dot+1:]
			for _, alias := range []string{ns + "__" + member, ns + "=" + member, ns + "_" + member, member} {
				add(alias, name)
			}
		}
		if strings.Contains(name, "__") {
			add(strings.ReplaceAll(name, "__", "_"), name)
		}
	}
	for alias, names := range candidates {
		if _, exact := p.schemas[alias]; exact || len(names) != 1 {
			continue
		}
		for name := range names {
			p.aliases[alias] = name
		}
	}
	p.resetScan()
	return p, nil
}

// Status returns the state and diagnostic counters exposed by Python's parser.
// Clean is true when no text is buffered, even if reasoning calls are pending.
type Status struct {
	State       string
	Clean       bool
	Pending     int
	Unread      int
	UnreadNames []string
	Rescued     int
	Refused     int
}

func (p *Parser) Status() Status {
	clean := p.buf == ""
	if p.state == "call" {
		clean = len(p.callBuf) == 0
	} else if p.state == "rcall" {
		clean = p.closed
	} else if p.opener != nil {
		clean = false
	}
	return Status{p.state, clean, len(p.pending), p.unread, append([]string(nil), p.unreadNames...), p.rescued, p.refused}
}

func space(r rune) bool      { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }
func lstrip(s string) string { return strings.TrimLeftFunc(s, space) }
func strip(s string) string  { return strings.TrimFunc(s, space) }

func hold(text string, tags ...string) int {
	best := 0
	for _, tag := range tags {
		for n := min(len(text), len(tag)-1); n > best; n-- {
			if strings.HasSuffix(text, tag[:n]) {
				best = n
				break
			}
		}
	}
	return best
}

func (p *Parser) track(text string) string {
	parts := strings.Split(text, "\n")
	for i, part := range parts {
		if i == len(parts)-1 {
			p.line = append(p.line, part...)
			break
		}
		p.line = append(p.line, part...)
		line := string(p.line)
		p.line = p.line[:0]
		s := lstrip(line)
		if p.fence != 0 {
			if strings.HasPrefix(s, strings.Repeat(string(p.fence), 3)) {
				p.fence = 0
			}
		} else if (strings.HasPrefix(s, "```") || strings.HasPrefix(s, "~~~")) &&
			!strings.Contains(s[3:], s[:3]) {
			p.fence = s[0]
		} else if s == "" {
			p.ticks = 0
		} else {
			p.ticks += strings.Count(line, "`")
		}
	}
	return text
}

func (p *Parser) openerOK() bool {
	return p.fence == 0 && strip(string(p.line)) == "" && p.ticks%2 == 0
}

func (p *Parser) inCodeAt(text string) bool {
	fence, ticks := p.fence, p.ticks
	line := append([]byte(nil), p.line...)
	p.track(text)
	result := p.fence != 0 || (p.ticks+strings.Count(string(p.line), "`"))%2 != 0
	p.fence, p.ticks, p.line = fence, ticks, line
	return result
}

func (p *Parser) quotedThinkEnd(i int) (quoted, wait bool) {
	var before byte
	if i > 0 {
		before = p.buf[i-1]
	} else if len(p.line) > 0 {
		before = p.line[len(p.line)-1]
	}
	if before != '"' && before != '\'' && before != '`' {
		return false, false
	}
	after := p.buf[i+len(thinkEnd):]
	if after == "" {
		return false, true
	}
	return after[0] != '\r' && after[0] != '\n', false
}

func (p *Parser) canonical(name string) string {
	if actual, ok := p.aliases[name]; ok {
		return actual
	}
	return name
}

func (p *Parser) callable(name string) bool {
	if len(p.schemas) == 0 {
		return !p.declared
	}
	_, ok := p.schemas[p.canonical(name)]
	return ok
}

// loose returns a normalized opener, "wait", or "".
func (p *Parser) loose(after string) (name string, end int, wait bool) {
	if m := looseOpen.FindStringSubmatchIndex(after); m != nil {
		name = after[m[2]:m[3]]
		if p.callable(name) {
			return name, m[1], false
		}
		return "", 0, false
	}
	if strings.ContainsAny(after, ">\n") || !strings.HasPrefix(after, "<") || !loosePart.MatchString(after) {
		return "", 0, false
	}
	rest := lstrip(after[1:])
	bare := strip(strings.TrimLeft(strip(after[1:]), "="))
	if strings.HasPrefix(rest, "=") || strings.HasPrefix(rest, "function") {
		return "", 0, true
	}
	for k := range p.schemas {
		if strings.HasPrefix(k, bare) {
			return "", 0, true
		}
	}
	for k := range p.aliases {
		if strings.HasPrefix(k, bare) {
			return "", 0, true
		}
	}
	return "", 0, false
}

func (p *Parser) newCall(name string, args map[string]any) (*ToolCall, error) {
	id, err := p.id()
	if err != nil {
		return nil, fmt.Errorf("output parser: generate call ID: %w", err)
	}
	if id == "" || !utf8.ValidString(id) {
		return nil, errors.New("output parser: ID generator returned an empty or invalid UTF-8 ID")
	}
	return &ToolCall{Name: name, Arguments: args, ID: id}, nil
}

func (p *Parser) release(deliver bool) []Event {
	var out []Event
	for _, item := range p.pending {
		if item.call != nil && deliver {
			out = append(out, Event{Kind: "tool_call", Call: item.call})
			p.rescued++
		} else {
			if item.call != nil {
				p.refused++
			}
			if item.raw != "" || len(item.whitespace) > 0 {
				out = append(out, Event{Kind: "reasoning", Text: p.track(item.raw + string(item.whitespace))})
			}
		}
	}
	p.pending = nil
	return out
}

func (p *Parser) fail(out []Event, err error) ([]Event, error) {
	p.failure = err
	return out, err
}

func (p *Parser) check() error {
	if p.failure != nil {
		return fmt.Errorf("%w: %v", ErrClosed, p.failure)
	}
	if p.closed {
		return ErrClosed
	}
	return nil
}

func (p *Parser) appendCall(text string) {
	var work *int
	if p.work != nil {
		work = &p.work.buffer
	}
	p.callBuf = appendTracked(p.callBuf, text, work)
	if p.state == "rcall" {
		p.callRunes += utf8.RuneCountInString(text)
	}
}

func (p *Parser) startCall(body, state string) {
	p.state, p.buf = state, ""
	p.callBuf, p.callRunes = nil, 0
	p.frame = newFrameScanner()
	p.thinkSearch = newNeedle(0)
	p.appendCall(body)
}

func (p *Parser) frameWork() *int {
	if p.work == nil {
		return nil
	}
	return &p.work.frame
}

// Feed consumes one valid UTF-8 string and returns all events produced by that
// feed. On error, events already produced in that feed precede the error; the
// parser is terminal and must not be resumed.
func (p *Parser) Feed(delta string) ([]Event, error) {
	if err := p.check(); err != nil {
		return nil, err
	}
	if !utf8.ValidString(delta) {
		return p.fail(nil, ErrInvalidUTF8)
	}
	if p.opener != nil {
		p.opener.buf = appendTracked(p.opener.buf, delta, p.openerCopyWork())
		if !p.opener.advance(p.openerScanWork()) {
			return nil, nil
		}
		p.buf, p.opener = string(p.opener.buf), nil
	} else if p.state == "call" || p.state == "rcall" {
		p.appendCall(delta)
	} else {
		if p.work != nil && p.buf != "" && delta != "" {
			p.work.heldStringCopy += len(p.buf) + len(delta)
		}
		p.buf += delta
	}
	var out []Event
	for {
		switch p.state {
		case "rcall":
			body := p.callBuf
			p.frame.process(body, p.frameWork())
			end := p.frame.end
			think := p.thinkSearch.find(body, thinkEnd, p.frameWork())
			if end >= 0 && (think < 0 || think >= end) {
				var call *ToolCall
				trimmed := strip(string(body[:end]))
				name := ""
				if len(trimmed) >= len(toolparser.FuncStart) {
					name = strings.SplitN(trimmed[len(toolparser.FuncStart):], ">", 2)[0]
				}
				name = p.canonical(strip(name))
				if schema, ok := p.schemas[name]; ok {
					parsed, err := toolparser.ParseToolCall(string(body[:end]), schema)
					if errors.Is(err, toolparser.ErrUnsupportedUnicode) {
						return p.fail(out, err)
					}
					if err == nil {
						call, err = p.newCall(name, parsed.Arguments)
						if err != nil {
							return p.fail(out, err)
						}
					}
				}
				raw := callStart + string(body[:end+len(callEnd)])
				if call != nil || len(p.pending) > 0 {
					p.pending = append(p.pending, pendingCall{raw: raw, call: call})
				} else {
					out = append(out, Event{Kind: "reasoning", Text: p.track(raw)})
				}
				p.buf, p.state = string(body[end+len(callEnd):]), "reasoning"
				p.callBuf = nil
			} else if think >= 0 {
				out = append(out, p.release(false)...)
				out = append(out, Event{Kind: "reasoning", Text: p.track(callStart + string(body[:think]))})
				p.buf, p.state = string(body[think:]), "reasoning"
				p.callBuf = nil
			} else if p.callRunes > rcallMax {
				out = append(out, p.release(false)...)
				out = append(out, Event{Kind: "reasoning", Text: p.track(callStart + string(body))})
				p.buf, p.state = "", "reasoning"
				p.callBuf = nil
			} else {
				return out, nil
			}
		case "reasoning":
			if len(p.pending) > 0 {
				stripped := lstrip(p.buf)
				if len(stripped) < len(p.buf) {
					last := len(p.pending) - 1
					var work *int
					if p.work != nil {
						work = &p.work.pendingCopy
					}
					p.pending[last].whitespace = appendTracked(p.pending[last].whitespace, p.buf[:len(p.buf)-len(stripped)], work)
					p.buf = stripped
				}
				if p.buf == "" {
					return out, nil
				}
				if strings.HasPrefix(p.buf, callStart) {
					p.startCall(p.buf[len(callStart):], "rcall")
				} else if strings.HasPrefix(p.buf, thinkEnd) {
					out = append(out, p.release(true)...)
				} else if strings.HasPrefix(callStart, p.buf) || strings.HasPrefix(thinkEnd, p.buf) {
					return out, nil
				} else {
					out = append(out, p.release(false)...)
				}
				continue
			}
			i, tool := strings.Index(p.buf, thinkEnd), -1
			if len(p.schemas) > 0 {
				tool = strings.Index(p.buf, callStart)
			}
			if tool >= 0 && (i < 0 || tool < i) {
				if tool > 0 {
					out = append(out, Event{Kind: "reasoning", Text: p.track(p.buf[:tool])})
				}
				if p.openerOK() {
					p.startCall(p.buf[tool+len(callStart):], "rcall")
				} else {
					out = append(out, Event{Kind: "reasoning", Text: p.track(callStart)})
					p.buf = p.buf[tool+len(callStart):]
				}
				continue
			}
			if i < 0 {
				tags := []string{thinkEnd}
				if len(p.schemas) > 0 {
					tags = append(tags, callStart)
				}
				keep := hold(p.buf, tags...)
				if len(p.buf) > keep {
					out = append(out, Event{Kind: "reasoning", Text: p.track(p.buf[:len(p.buf)-keep])})
					p.buf = p.buf[len(p.buf)-keep:]
				}
				return out, nil
			}
			quoted, wait := p.quotedThinkEnd(i)
			if wait {
				if i > 0 {
					out = append(out, Event{Kind: "reasoning", Text: p.track(p.buf[:i])})
					p.buf = p.buf[i:]
				}
				return out, nil
			}
			if quoted {
				out = append(out, Event{Kind: "reasoning", Text: p.track(p.buf[:i+len(thinkEnd)])})
				p.buf = p.buf[i+len(thinkEnd):]
				continue
			}
			if i > 0 {
				out = append(out, Event{Kind: "reasoning", Text: p.track(p.buf[:i])})
			}
			p.buf, p.state, p.lead = p.buf[i+len(thinkEnd):], "content", true
			p.fence, p.line, p.ticks = 0, nil, 0
		case "content":
			if p.lead {
				stripped := strings.TrimLeft(p.buf, "\n")
				if stripped == "" {
					p.buf = ""
					return out, nil
				}
				p.buf, p.lead = stripped, false
			}
			i := strings.Index(p.buf, callStart)
			if i < 0 {
				j := len(p.buf) - hold(p.buf, callStart)
				for j > 0 && p.buf[j-1] == '\n' {
					j--
				}
				if j > 0 {
					out = append(out, Event{Kind: "content", Text: p.track(p.buf[:j])})
					p.buf = p.buf[j:]
				}
				return out, nil
			}
			if p.inCodeAt(p.buf[:i]) {
				out = append(out, Event{Kind: "content", Text: p.track(p.buf[:i+len(callStart)])})
				p.buf = p.buf[i+len(callStart):]
				continue
			}
			after := lstrip(p.buf[i+len(callStart):])
			if len(p.schemas) > 0 && !strings.HasPrefix(after, toolparser.FuncStart) {
				name, end, wait := p.loose(after)
				if wait {
					out = p.holdBefore(out, i)
					return out, nil
				}
				if name != "" {
					at := len(p.buf) - len(after)
					p.buf = p.buf[:at] + toolparser.FuncStart + name + ">" + after[end:]
					continue
				}
			}
			if strings.HasPrefix(after, "<") && !strings.HasPrefix(toolparser.FuncStart, after) && !strings.HasPrefix(after, toolparser.FuncStart) {
				p.unread++
			}
			match := functionOpen.FindStringSubmatch(after)
			if match != nil {
				raw := match[1]
				name := p.canonical(strip(raw))
				if raw != strip(raw) {
					if _, ok := p.schemas[name]; !ok {
						match = nil
					}
				}
			}
			partial := strings.HasPrefix(toolparser.FuncStart, after) || functionPart.MatchString(after)
			if match == nil && !partial {
				out = append(out, Event{Kind: "content", Text: p.track(p.buf[:i+len(callStart)])})
				p.buf = p.buf[i+len(callStart):]
				continue
			}
			if match == nil {
				out = p.holdBefore(out, i)
				p.holdOpener(strings.Index(p.buf, callStart))
				return out, nil
			}
			rawName := strip(match[1])
			if (len(p.schemas) > 0 || p.declared) && !p.callable(rawName) {
				p.unread++
				if len(p.unreadNames) < 5 {
					runes := []rune(rawName)
					p.unreadNames = append(p.unreadNames, string(runes[:min(80, len(runes))]))
				}
				out = append(out, Event{Kind: "content", Text: p.track(p.buf[:i+len(callStart)])})
				p.buf = p.buf[i+len(callStart):]
				continue
			}
			if i > 0 && strip(p.buf[:i]) != "" {
				out = append(out, Event{Kind: "content", Text: p.track(strings.TrimRight(p.buf[:i], "\n"))})
			}
			p.startCall(p.buf[i+len(callStart):], "call")
		case "call":
			p.frame.process(p.callBuf, p.frameWork())
			i := p.frame.end
			drop := len(callEnd)
			if p.frame.functionEnd >= 0 && p.frame.next.process(p.callBuf, p.frameWork()) {
				i, drop = p.frame.functionEnd, 0
			}
			if p.stream {
				body := p.callBuf
				if i >= 0 {
					body = body[:i]
				}
				events, err := p.scanCall(body)
				out = append(out, events...)
				if err == nil && i >= 0 {
					out = append(out, p.closeScan(body)...)
				}
				if err != nil {
					return p.fail(out, err)
				}
			}
			if i < 0 {
				return out, nil
			}
			body := string(p.callBuf[:i])
			p.buf = string(p.callBuf[i+drop:])
			p.callBuf = nil
			event, err := p.finishCall(body)
			if err != nil {
				return p.fail(out, err)
			}
			out = append(out, event)
		default:
			panic("invalid output parser state")
		}
	}
}

func (p *Parser) holdBefore(out []Event, i int) []Event {
	j := i
	for j > 0 && p.buf[j-1] == '\n' {
		j--
	}
	if j > 0 {
		out = append(out, Event{Kind: "content", Text: p.track(p.buf[:j])})
		p.buf = p.buf[j:]
	}
	return out
}

func (p *Parser) finishCall(body string) (Event, error) {
	trimmed := strip(body)
	raw := ""
	if len(trimmed) >= len(toolparser.FuncStart) {
		raw = strings.SplitN(trimmed[len(toolparser.FuncStart):], ">", 2)[0]
	}
	name := p.canonical(strip(raw))
	if raw != strip(raw) {
		if _, ok := p.schemas[name]; !ok {
			return Event{}, errors.New("whitespace tool name is not declared")
		}
	}
	parsed, err := toolparser.ParseToolCall(body, p.schemas[name])
	if err != nil {
		return Event{}, err
	}
	var call *ToolCall
	if p.scan.call != nil {
		call = &ToolCall{Name: name, Arguments: parsed.Arguments, ID: p.scan.call.ID}
	} else {
		call, err = p.newCall(name, parsed.Arguments)
		if err != nil {
			return Event{}, err
		}
	}
	p.resetScan()
	p.state, p.lead = "content", true
	return Event{Kind: "tool_call", Call: call}, nil
}

// Finish ends a turn. Empty reason is Python's None; "stop" is also natural.
func (p *Parser) Finish(reason string) ([]Event, error) {
	if err := p.check(); err != nil {
		return nil, err
	}
	p.closed = true
	var out []Event
	if p.opener != nil {
		p.buf, p.opener = string(p.opener.buf), nil
	}
	if len(p.pending) > 0 {
		out = append(out, p.release((reason == "" || reason == "stop") && p.state == "reasoning" && p.buf == "")...)
	}
	if p.state == "call" && p.stream && p.scan.call != nil {
		events, err := p.scanCall(p.callBuf)
		out = append(out, events...)
		if err != nil {
			return p.fail(out, err)
		}
		if p.scan.state == "function_done" && strip(string(p.callBuf[p.scan.pos:])) == "" && (reason == "" || reason == "stop") {
			out = append(out, p.closeScan(p.callBuf)...)
			out = append(out, Event{Kind: "tool_call", Call: p.scan.call})
		} else if p.scan.state == "function_done" {
			out = append(out, Event{Kind: "content", Text: p.track(callStart + string(p.callBuf))})
		}
		p.callBuf = nil
		p.resetScan()
		return out, nil
	}
	if p.state == "reasoning" && p.buf == thinkEnd {
		p.buf, p.state = "", "content"
	}
	if p.state == "call" || p.state == "rcall" {
		if p.state == "rcall" {
			out = append(out, Event{Kind: "reasoning", Text: callStart + string(p.callBuf)})
		} else if len(p.callBuf) > 0 {
			out = append(out, Event{Kind: "content", Text: callStart + string(p.callBuf)})
		}
		p.callBuf = nil
	} else if p.buf != "" {
		kind, text := "content", p.buf
		if p.state == "reasoning" {
			kind = "reasoning"
		}
		out = append(out, Event{Kind: kind, Text: text})
		p.buf = ""
	}
	return out, nil
}
