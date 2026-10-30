package outputparser

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"strata/toolparser"
)

type specimen struct {
	Name   string   `json:"name"`
	Input  string   `json:"input"`
	Tools  []string `json:"tools"`
	Finish string   `json:"finish"`
}

type fixtureSet struct {
	Call      string           `json:"call"`
	Tools     []map[string]any `json:"tools"`
	Specimens []specimen       `json:"specimens"`
}

type request struct {
	Parts        []string         `json:"parts"`
	Thinking     bool             `json:"thinking"`
	Tools        []map[string]any `json:"tools"`
	Stream       bool             `json:"stream"`
	DeclaredOnly bool             `json:"declared_only"`
	Finish       string           `json:"finish"`
}

func tree(v any) any {
	switch x := v.(type) {
	case string:
		points := make([]int, 0, len(x))
		for _, r := range x {
			points = append(points, int(r))
		}
		return map[string]any{"string": points}
	case json.Number:
		return map[string]any{"integer": string(x)}
	case float64:
		return map[string]any{"float_bits": fmt.Sprintf("%016x", math.Float64bits(x))}
	case []any:
		result := make([]any, 0, len(x))
		for _, item := range x {
			result = append(result, tree(item))
		}
		return result
	case map[string]any:
		keys := make([]string, 0, len(x))
		for key := range x {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		items := make([]any, 0, len(x))
		for _, key := range keys {
			items = append(items, []any{tree(key), tree(x[key])})
		}
		return map[string]any{"object": items}
	default:
		return v
	}
}

func runGo(r request) (any, error) {
	n := 0
	p, err := New(Options{Thinking: r.Thinking, Tools: r.Tools, StreamTools: r.Stream,
		DeclaredOnly: r.DeclaredOnly, ID: func() (string, error) {
			n++
			return fmt.Sprintf("call_%d", n), nil
		}})
	if err != nil {
		return nil, err
	}
	ids := map[string]int{}
	norm := func(events []Event) []any {
		out := make([]any, 0, len(events))
		for _, event := range events {
			var call any
			if event.Call != nil {
				id, ok := ids[event.Call.ID]
				if !ok {
					id = len(ids) + 1
					ids[event.Call.ID] = id
				}
				call = map[string]any{"name": event.Call.Name, "arguments": tree(event.Call.Arguments), "id": id}
			}
			out = append(out, map[string]any{"kind": event.Kind, "text": event.Text, "call": call})
		}
		return out
	}
	status := func() any {
		s := p.Status()
		names := append([]string{}, s.UnreadNames...)
		return map[string]any{"state": s.State, "clean": s.Clean, "pending": s.Pending, "unread": s.Unread,
			"unread_names": names, "rescued": s.Rescued, "refused": s.Refused}
	}
	batches, states := []any{}, []any{}
	for _, part := range r.Parts {
		events, err := p.Feed(part)
		if err != nil {
			return nil, err
		}
		batches = append(batches, norm(events))
		states = append(states, status())
	}
	events, err := p.Finish(r.Finish)
	if err != nil {
		return nil, err
	}
	batches = append(batches, norm(events))
	states = append(states, status())
	return map[string]any{"batches": batches, "states": states}, nil
}

func python(t *testing.T) string {
	t.Helper()
	path := filepath.Join("..", "..", ".venv", "bin", "python")
	if _, err := os.Stat(path); err != nil {
		t.Skip("project .venv Python unavailable")
	}
	return path
}

func loadFixtures(t *testing.T, interpreter string) fixtureSet {
	t.Helper()
	cmd := exec.Command(interpreter, "oracle.py")
	cmd.Stdin = strings.NewReader("{\"op\":\"fixtures\"}\n")
	data, err := cmd.Output()
	if err != nil {
		t.Fatalf("fixture oracle: %v: %s", err, data)
	}
	var response struct {
		Result fixtureSet `json:"result"`
		Error  string     `json:"error"`
	}
	if err := json.Unmarshal(data, &response); err != nil || response.Error != "" {
		t.Fatalf("fixtures: %v %s", err, data)
	}
	return response.Result
}

func toolsFor(names []string) []map[string]any {
	out := make([]map[string]any, 0, len(names))
	for _, name := range names {
		out = append(out, map[string]any{"name": name})
	}
	return out
}

func addSplits(requests *[]request, base request, text string, every bool) {
	runes := []rune(text)
	if every {
		for i := 0; i <= len(runes); i++ {
			r := base
			r.Parts = []string{string(runes[:i]), string(runes[i:])}
			*requests = append(*requests, r)
		}
	}
	r := base
	r.Parts = []string{text}
	*requests = append(*requests, r)
	r.Parts = make([]string, 0, len(runes))
	for _, c := range runes {
		r.Parts = append(r.Parts, string(c))
	}
	*requests = append(*requests, r)
}

func TestPythonDifferential(t *testing.T) {
	interpreter := python(t)
	fixtures := loadFixtures(t, interpreter)
	var requests []request
	// All existing reasoning specimens are checked whole and character by
	// character; representative acts, quotes, fences and cuts at every split.
	allSplits := map[int]bool{0: true, 1: true, 2: true, 8: true, 12: true, 13: true, 14: true, 17: true, 19: true, 23: true}
	for i, sp := range fixtures.Specimens {
		base := request{Thinking: true, Tools: toolsFor(sp.Tools), Finish: sp.Finish}
		addSplits(&requests, base, sp.Input, allSplits[i])
	}
	for _, stream := range []bool{false, true} {
		for _, thinking := range []bool{false, true} {
			prefix := ""
			if thinking {
				prefix = "Inspect.\n</think>\n\n"
			}
			base := request{Thinking: thinking, Tools: fixtures.Tools, Stream: stream}
			addSplits(&requests, base, prefix+fixtures.Call+"\nDone", true)
			addSplits(&requests, base, prefix+"Quoted `"+fixtures.Call+"` and a literal <tool_call> example.", true)
			addSplits(&requests, base, prefix+"```xml\n"+fixtures.Call+"\n```\nDone.", true)
		}
	}
	tools := []map[string]any{
		{"name": "scope.write", "parameters": map[string]any{"properties": map[string]any{
			"text": map[string]any{"type": "string"}, "n": map[string]any{"type": "integer"},
		}}},
	}
	for _, name := range []string{"scope.write", "scope__write", "scope=write", "scope_write", "write", "missing", " write "} {
		for _, stream := range []bool{false, true} {
			text := "<tool_call><function=" + name + "><parameter=text>\nAé🙂\n</parameter>" +
				"<parameter=n>12</parameter></function></tool_call>End"
			addSplits(&requests, request{Tools: tools, Stream: stream}, text, true)
		}
	}
	for _, text := range []string{
		"quoted `</think>` again</think>\nanswer",
		"hello\n<tool_",
		"<tool_call><function=missing></function></tool_call>",
		"<tool_call><=scope.write><parameter=text>abc</parameter></function></tool_call>",
		"<tool_call><function=scope.write><parameter=text>literal </parameter> text </tool_call> end</parameter></function></tool_call>",
		"<tool_call><function=scope.write></function><tool_call><function=scope.write></function></tool_call>",
		"<tool_call><function=scope.write><parameter=text>unfinished",
		"<tool_call><function=scope.write></function>",
	} {
		for _, stream := range []bool{false, true} {
			addSplits(&requests, request{Tools: tools, Thinking: strings.HasPrefix(text, "quoted"), Stream: stream}, text, true)
		}
	}
	for _, declared := range []bool{false, true} {
		addSplits(&requests, request{DeclaredOnly: declared}, "<tool_call><function=f></function></tool_call>", true)
	}
	for _, aliases := range [][]map[string]any{
		{{"name": "scope.write"}, {"name": "scope__write", "parameters": map[string]any{"properties": map[string]any{"n": map[string]any{"type": "string"}}}}},
		{{"name": "a.b__c"}, {"name": "a__b.c"}},
		{{"name": "mcp__playwright__browser_navigate"}},
	} {
		for _, name := range []string{"scope__write", "a__b__c", "mcp_playwright_browser_navigate"} {
			text := "<tool_call><function=" + name + "><parameter=n>123</parameter></function></tool_call>"
			addSplits(&requests, request{Tools: aliases, Stream: true}, text, true)
		}
	}
	for _, value := range []string{`false`, `null`, `-32`, `["a", 42, true, null]`, `{"a":[1,2]}`, `"é🙂"`} {
		text := "<tool_call><function=scope.write><parameter=n>" + value + "</parameter></function></tool_call>"
		addSplits(&requests, request{Tools: tools, Stream: true}, text, true)
	}
	for _, value := range []string{`1.5`, `-0.0`, `[NaN, Infinity, -Infinity, 1e400]`,
		`{"b":1,"a":2}`, `{"duplicate":1,"duplicate":2}`, `not json`, `123456789012345678901234567890`} {
		text := "<tool_call><function=scope.write><parameter=n>" + value + "</parameter></function></tool_call>"
		addSplits(&requests, request{Tools: tools}, text, true)
	}
	for _, text := range []string{
		"<tool_call><function=scope.write><parameter=text>\x00\x7f\t\"\\é🙂</parameter></function></tool_call>",
		"Need two calls.\n<tool_call><function=scope.write></function></tool_call>\n<tool_call><function=scope.write></function></tool_call></think>Done",
		"<tool_call><function=scope.write><parameter=text>one </parameter> x </parameter> \u2003 </function></tool_call>End",
		"<tool_call><function=scope.write><parameter=text>one </parameter> x </parameter>    <parameter=n>2</parameter></function></tool_call>",
		"<tool_call><function=scope.write><parameter=n>one </parameter> x </parameter> \n</function></tool_call>End",
		"<tool_call><function=scope.write></function> \u2003 <tool_call>  <function=scope.write></function></tool_call>End",
	} {
		for _, stream := range []bool{false, true} {
			addSplits(&requests, request{Tools: tools, Thinking: strings.HasPrefix(text, "Need"), Stream: stream}, text, true)
		}
	}
	for _, finish := range []string{"stop", "length", "error", "cancel"} {
		for _, stream := range []bool{false, true} {
			for _, text := range []string{
				"<tool_call><function=scope.write><parameter=text>unfinished",
				"<tool_call><function=scope.write></function> ",
				"before\n<tool_call><function=scope.write><parameter=text>unterminated",
			} {
				addSplits(&requests, request{Tools: tools, Thinking: strings.HasPrefix(text, "before"), Stream: stream, Finish: finish}, text, true)
			}
		}
	}
	for _, size := range []int{32768, 32769} {
		text := "<tool_call>" + strings.Repeat("🙂", size)
		requests = append(requests, request{Thinking: true, Tools: tools, Parts: []string{text}})
		parts := []string{"<tool_call>"}
		for remaining := size; remaining > 0; remaining -= min(512, remaining) {
			parts = append(parts, strings.Repeat("🙂", min(512, remaining)))
		}
		requests = append(requests, request{Thinking: true, Tools: tools, Parts: parts})
	}
	// Exercise a held first opener and pending reasoning whitespace after every
	// input character, including valid, invalid, cut-off and Unicode cases.
	openerNames := []string{strings.Repeat("x", 128), "é🙂"}
	for _, name := range openerNames {
		for _, declared := range []bool{false, true} {
			var declaredTools []map[string]any
			if declared {
				declaredTools = []map[string]any{{"name": name}}
			}
			for _, stream := range []bool{false, true} {
				for _, suffix := range []string{
					"></function></tool_call>done", " x", "", " \t></function></tool_call>",
				} {
					text := "prefix\n<tool_call><function=" + name + suffix
					addSplits(&requests, request{Tools: declaredTools, Stream: stream}, text, true)
				}
			}
		}
	}
	for _, stream := range []bool{false, true} {
		for _, after := range []string{
			"<function=f></function></tool_call>done", "not a call",
			"<function=f x", "<=f></function></tool_call>", "",
		} {
			text := "prefix\n<tool_call>" + strings.Repeat(" ", 64) + "\u2003\n" + after
			addSplits(&requests, request{Tools: toolsFor([]string{"f"}), Stream: stream}, text, true)
		}
	}
	call := "<tool_call><function=f></function></tool_call>"
	for _, finish := range []string{"stop", "length"} {
		for _, suffix := range []string{"", "more", "</think>done", call + "</think>done", "<tool_c"} {
			text := "start\n" + call + strings.Repeat(" ", 128) + "\u2003\n" + suffix
			addSplits(&requests, request{Thinking: true, Tools: toolsFor([]string{"f"}), Finish: finish}, text, true)
		}
	}
	cmd := exec.Command(interpreter, "oracle.py")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start oracle: %v", err)
	}
	defer func() {
		stdin.Close()
		if cmd.ProcessState == nil {
			cmd.Process.Kill()
			cmd.Wait()
		}
	}()
	encoder := json.NewEncoder(stdin)
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	for i, r := range requests {
		if err := encoder.Encode(r); err != nil {
			t.Fatalf("send request %d: %v", i, err)
		}
		if !scanner.Scan() {
			t.Fatalf("oracle stopped at %d: %v, stderr %s", i, scanner.Err(), stderr.String())
		}
		var response struct {
			Result  any    `json:"result"`
			Error   string `json:"error"`
			Message string `json:"message"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		got, err := runGo(r)
		if err != nil || response.Error != "" {
			t.Fatalf("request %d %#v: Go error %v, oracle %s %s", i, r, err, response.Error, response.Message)
		}
		encoded, err := json.Marshal(got)
		if err != nil {
			t.Fatal(err)
		}
		var normalized any
		if err := json.Unmarshal(encoded, &normalized); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(normalized, response.Result) {
			want, _ := json.Marshal(response.Result)
			t.Fatalf("request %d %#v:\nGo %s\nPython %s", i, r, encoded, want)
		}
	}
	if err := stdin.Close(); err != nil {
		t.Fatalf("close oracle input: %v", err)
	}
	if scanner.Scan() {
		t.Fatal("extra oracle response")
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("oracle output: %v", err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("oracle: %v: %s", err, stderr.String())
	}
	t.Logf("compared %d complete parser traces", len(requests))
}

func TestExplicitFailuresAndLifecycle(t *testing.T) {
	if _, err := New(Options{Recover: true}); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("recovery: %v", err)
	}
	if _, err := New(Options{Tools: []map[string]any{{"name": 3}}}); err == nil {
		t.Fatal("non-string tool name accepted")
	}
	for _, schema := range []map[string]any{
		{"name": "x", "parameters": []any{}},
		{"name": "x", "parameters": map[string]any{"properties": "invalid"}},
		{"name": "x", "parameters": map[string]any{"properties": map[string]any{"x": "invalid"}}},
	} {
		if _, err := New(Options{Tools: []map[string]any{schema}}); err == nil {
			t.Fatalf("invalid schema accepted: %#v", schema)
		}
	}
	p, err := New(Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Feed(string([]byte{0xff})); !errors.Is(err, ErrInvalidUTF8) {
		t.Fatalf("UTF-8: %v", err)
	}
	if _, err := p.Finish(""); !errors.Is(err, ErrClosed) {
		t.Fatalf("failed parser resumed: %v", err)
	}
	p, err = New(Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Finish("stop"); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Feed("later"); !errors.Is(err, ErrClosed) {
		t.Fatalf("finished parser resumed: %v", err)
	}
	for _, value := range []string{`1.5`, `{"b": 1, "a": 2}`, `[0, {"b":1,"a":2}]`} {
		p, err = New(Options{StreamTools: true})
		if err != nil {
			t.Fatal(err)
		}
		events, err := p.Feed("<tool_call><function=f><parameter=x>" + value + "</parameter></function></tool_call>")
		if !errors.Is(err, ErrUnsupported) || len(events) == 0 || events[0].Kind != "tool_start" {
			t.Fatalf("unsupported value %s: events %#v error %v", value, events, err)
		}
		if _, err := p.Finish(""); !errors.Is(err, ErrClosed) {
			t.Fatalf("unsupported value resumed: %v", err)
		}
	}
	p, err = New(Options{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Feed(`<tool_call><function=f><parameter=x>"\ud800"</parameter></function></tool_call>`)
	if !errors.Is(err, toolparser.ErrUnsupportedUnicode) {
		t.Fatalf("unpaired surrogate: %v", err)
	}
}

func TestReplayAndStatus(t *testing.T) {
	tools := []map[string]any{{"name": "f"}}
	text := []string{"go\n", "<tool_call><function=f>", "</function></tool_call>", "\n", "</think>done"}
	replay := func() ([]Event, []Status) {
		id := 0
		p, err := New(Options{Thinking: true, Tools: tools, StreamTools: true, ID: func() (string, error) {
			id++
			return fmt.Sprintf("replay_%d", id), nil
		}})
		if err != nil {
			t.Fatal(err)
		}
		var events []Event
		var statuses []Status
		for _, part := range text {
			batch, err := p.Feed(part)
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, batch...)
			statuses = append(statuses, p.Status())
		}
		batch, err := p.Finish("stop")
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, batch...)
		return events, statuses
	}
	first, states := replay()
	second, _ := replay()
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("replay differs: %#v vs %#v", first, second)
	}
	if !states[2].Clean || states[2].Pending != 1 || states[3].Pending != 1 || states[4].Rescued != 1 {
		t.Fatalf("pending lifecycle: %#v", states)
	}
}

func TestMalformedCallAndIDFailure(t *testing.T) {
	p, err := New(Options{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Feed("<tool_call><function=f>junk<parameter=broken</tool_call>")
	if err == nil || !strings.Contains(err.Error(), "substring not found") {
		t.Fatalf("malformed call: %v", err)
	}
	boom := errors.New("ID failure")
	p, err = New(Options{ID: func() (string, error) { return "", boom }})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Feed("<tool_call><function=f></function></tool_call>")
	if !errors.Is(err, boom) {
		t.Fatalf("ID failure: %v", err)
	}
	if _, err := p.Feed(""); !errors.Is(err, ErrClosed) {
		t.Fatalf("failed parser resumed: %v", err)
	}
}
