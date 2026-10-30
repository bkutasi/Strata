package requestnorm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

type referenceCase struct {
	Name    string          `json:"-"`
	Family  Family          `json:"family"`
	Request json.RawMessage `json:"request"`
	Require bool            `json:"require,omitempty"`
}

func makeCase(name, request string) referenceCase {
	return referenceCase{Name: name, Family: FamilyQwen, Request: json.RawMessage(request)}
}

func encodeJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestOpenAIDifferential(t *testing.T) {
	python := filepath.Join("..", "..", ".venv", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Skip("project .venv Python unavailable")
	}
	cases := []referenceCase{
		makeCase("empty", `{}`),
		{Name: "empty required", Family: FamilyQwen, Request: json.RawMessage(`{"messages":[]}`), Require: true},
		makeCase("null messages", `{"messages":null}`),
		makeCase("basic and late system", `{"messages":[{"role":"developer","content":"system"},{"role":"user","content":"hello"},{"role":"system","content":"later"},{"role":"developer","content":"later dev"},{"role":"assistant","content":null,"reasoning_content":"thinking"}]}`),
		makeCase("plain parts and refusal", `{"messages":[{"role":"user","content":[{"text":"α"},{"type":"input_text","text":"b"},{"type":"text"},{"type":"refusal","refusal":"no"},{"type":"text","text":""}]}]}`),
		makeCase("mixed images", `{"messages":[{"role":"user","content":[{"type":"text"},{"type":"text","text":""},{"text":"see "},{"type":"image_url","image_url":{"url":"https://x"}},{"type":"input_image","image_url":"pic.png"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"AA=="}},{"type":"image","source":{"path":"file.png"}},{"type":"refusal","refusal":"end"}]}]}`),
		makeCase("image defaults", `{"messages":[{"role":"assistant","content":[{"type":"image","source":{"type":"base64","data":"a"}},{"type":"image_url"},{"type":"image"}]}]}`),
		makeCase("tool history reversed", `{"messages":[{"role":"user","content":"go"},{"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}},{"id":"b","type":"function","function":{"name":"read","arguments":{"path":"b"}}}]},{"role":"tool","tool_call_id":"b","content":[{"type":"text","text":"B"},{"type":"image_url","image_url":"b.png"}]},{"role":"tool","tool_call_id":"a","content":"A"}]}`),
		makeCase("partial result run", `{"messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f"}},{"id":"b","function":{"name":"g"}}]},{"role":"tool","tool_call_id":"b","content":"B"}]}`),
		makeCase("duplicate call ids", `{"messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f"}},{"id":"a","function":{"name":"g"}}]},{"role":"tool","tool_call_id":"a","content":"first"},{"role":"tool","tool_call_id":"a","content":"second"}]}`),
		makeCase("duplicate result ids", `{"messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f"}},{"id":"b","function":{"name":"g"}}]},{"role":"tool","tool_call_id":"b","content":"first"},{"role":"tool","tool_call_id":"b","content":"second"}]}`),
		makeCase("intervening message", `{"messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f"}},{"id":"b","function":{"name":"g"}}]},{"role":"user","content":"intervening"},{"role":"tool","tool_call_id":"b","content":"B"},{"role":"tool","tool_call_id":"a","content":"A"}]}`),
		makeCase("arguments", `{"messages":[{"role":"assistant","tool_calls":[{"name":"a"},{"name":"b","arguments":" "},{"name":"c","arguments":"{\"z\":1,\"a\":{\"y\":2,\"b\":3}}"},{"name":"d","arguments":"{cut"},{"name":"e","arguments":"[1,2]"},{"name":"f","arguments":[1,{"x":"你好","v":true},null]},{"name":"g","arguments":-0},{"name":"h","arguments":true}]}]}`),
		makeCase("schema types and order", `{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"description":"desc","name":"ordered_schema","parameters":{"type":"object","properties":{"z":{"type":"string","enum":["a","b"]},"a":{"type":"array","items":{"type":"number"}},"b":{"type":["boolean","null"]},"n":{"type":"integer","default":1},"o":{"$ref":"#/$defs/o"}},"required":["z","a"],"additionalProperties":false,"$defs":{"o":{"type":"object","properties":{"v":{"type":"number","default":1.5}}}}},"strict":true}}]}`),
		makeCase("bare tool without schema", `{"messages":[{"role":"user","content":"x"}],"tools":[{"name":"bare","description":"é","parameters":null}]}`),
		makeCase("cache first", `{"tools":[{"name":"cache_b","parameters":{"z":1,"a":2}},{"name":"cache_a","parameters":{"a":1,"z":2}}]}`),
		makeCase("cache reversed with updated metadata", `{"tools":[{"name":"cache_a","description":"new a"},{"name":"cache_b","description":"new b"}]}`),
		makeCase("duplicates not reordered", `{"tools":[{"name":"duplicate","description":"first"},{"name":"duplicate","description":"second"}]}`),
		makeCase("duplicate reversed not cached", `{"tools":[{"name":"duplicate","description":"second"},{"name":"duplicate","description":"first"}]}`),
		makeCase("reasoning off false", `{"reasoning_effort":false,"reasoning":{"effort":"high"}}`),
		makeCase("reasoning fallback", `{"reasoning_effort":"","reasoning":{"effort":"HIGH"}}`),
		makeCase("reasoning option override", `{"reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"medium"}}`),
		makeCase("thinking option stops subsequent effort", `{"chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"bad"}}`),
		makeCase("effort option errors before thinking disabled", `{"chat_template_kwargs":{"reasoning_effort":"bad","enable_thinking":false}}`),
		makeCase("thinking option clears effort", `{"reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"low","enable_thinking":false}}`),
		makeCase("bad messages", `{"messages":"not json"}`),
		makeCase("nonobjects in messages", `{"messages":[{"role":"user"},7]}`),
		makeCase("bad calls", `{"messages":[{"role":"assistant","tool_calls":"{"}]}`),
		makeCase("nonobject calls", `{"messages":[{"role":"assistant","tool_calls":[1]}]}`),
		makeCase("bad call function", `{"messages":[{"role":"assistant","tool_calls":[{"function":"x"}]}]}`),
		makeCase("bad tools", `{"tools":"auto"}`),
		makeCase("missing tool name", `{"tools":[{"type":"function","function":{"parameters":{}}}]}`),
		makeCase("tool schema string", `{"tools":[{"name":"f","parameters":"x"}]}`),
		makeCase("tool schema list", `{"tools":[{"name":"f","parameters":[]}]}`),
		makeCase("tool schema bool", `{"tools":[{"name":"f","parameters":false}]}`),
		makeCase("tool schema float", `{"tools":[{"name":"f","parameters":1.5}]}`),
		makeCase("bad effort", `{"reasoning_effort":"extreme"}`),
		{Name: "deepseek high", Family: FamilyDeepSeek, Request: json.RawMessage(`{"reasoning_effort":"high"}`)},
		{Name: "deepseek max", Family: FamilyDeepSeek, Request: json.RawMessage(`{"reasoning_effort":"xhigh"}`)},
		{Name: "deepseek numeric", Family: FamilyDeepSeek, Request: json.RawMessage(`{"reasoning_effort":37}`)},
		{Name: "deepseek out of range", Family: FamilyDeepSeek, Request: json.RawMessage(`{"reasoning_effort":101}`)},
		{Name: "deepseek negative zero", Family: FamilyDeepSeek, Request: json.RawMessage(`{"reasoning_effort":-0}`)},
	}
	callArray := `[{"id":"id","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]`
	encodedMessages := `[{"role":"user","content":"hey"},{"role":"assistant","tool_calls":` + encodeJSON(t, callArray) + `}]`
	cases = append(cases, makeCase("double encoded messages and calls", `{"messages":`+encodeJSON(t, encodedMessages)+`}`))
	cases = append(cases, makeCase("double encoded tools", `{"tools":`+encodeJSON(t, `[{"type":"function","function":{"name":"encoded","parameters":{"type":"object"}}}]`)+`}`))

	// Populate 64 other tool sets, evicting the first set from the 64-entry LRU.
	cases = append(cases, makeCase("eviction seed", `{"tools":[{"name":"evict_b"},{"name":"evict_a"}]}`))
	for i := 0; i < 64; i++ {
		cases = append(cases, makeCase(fmt.Sprintf("fill cache %02d", i),
			fmt.Sprintf(`{"tools":[{"name":"fill%02d_a"},{"name":"fill%02d_b"}]}`, i, i)))
	}
	cases = append(cases, makeCase("eviction revisit", `{"tools":[{"name":"evict_a"},{"name":"evict_b"}]}`))

	var stdin bytes.Buffer
	for _, c := range cases {
		if err := json.NewEncoder(&stdin).Encode(c); err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
	}
	cmd := exec.Command(python, "oracle.py")
	cmd.Stdin = &stdin
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("reference failed: %v: %s", err, stderr.String())
	}
	norms := map[Family]*Normalizer{}
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for _, c := range cases {
		if !scanner.Scan() {
			t.Fatalf("%s: reference stopped: %v; stderr: %s", c.Name, scanner.Err(), stderr.String())
		}
		var ref struct {
			Result json.RawMessage `json:"result"`
			Error  string          `json:"error"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &ref); err != nil {
			t.Fatalf("%s: invalid reference response %q: %v", c.Name, scanner.Text(), err)
		}
		n := norms[c.Family]
		if n == nil {
			n, err = NewNormalizer(c.Family)
			if err != nil {
				t.Fatal(err)
			}
			norms[c.Family] = n
		}
		req, gotErr := DecodeOpenAI(c.Request)
		var result Result
		if gotErr == nil {
			result, gotErr = n.NormalizeOpenAI(req)
		}
		if gotErr == nil && c.Require {
			gotErr = RequireMessages(result.Messages)
		}
		if ref.Error != "" || gotErr != nil {
			// Match success versus error, without pinning human-facing prose.
			if (ref.Error == "") != (gotErr == nil) {
				t.Errorf("%s: reference error %q; Go error %v", c.Name, ref.Error, gotErr)
			}
			continue
		}
		gotJSON, err := json.Marshal(result)
		if err != nil {
			t.Fatalf("%s: marshal result: %v", c.Name, err)
		}
		got, err := ParseValue(gotJSON)
		if err != nil {
			t.Fatalf("%s: parse result: %v", c.Name, err)
		}
		want, err := ParseValue(ref.Result)
		if err != nil {
			t.Fatalf("%s: parse reference: %v", c.Name, err)
		}
		gotCanonical, _ := got.MarshalJSON()
		wantCanonical, _ := want.MarshalJSON()
		if !bytes.Equal(gotCanonical, wantCanonical) {
			t.Errorf("%s:\nGo:        %s\nreference: %s", c.Name, gotCanonical, wantCanonical)
		}
	}
	if scanner.Scan() {
		t.Fatalf("reference produced extra output: %s", scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestIDsSurviveWhileTemplateFormOmitsThem(t *testing.T) {
	req, err := DecodeOpenAI([]byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f"}},{"id":"b","function":{"name":"f"}}]},{"role":"tool","tool_call_id":"b","content":"B"},{"role":"tool","tool_call_id":"a","content":"A"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	n, _ := NewNormalizer(FamilyQwen)
	result, err := n.NormalizeOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Messages[0].ToolCalls[0].ID != "a" || result.Messages[1].ToolCallID != "a" ||
		result.Messages[2].ToolCallID != "b" || result.Messages[1].Content.Text != "A" {
		t.Fatalf("IDs or ordering lost: %#v", result.Messages)
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(`"id"`)) || bytes.Contains(data, []byte("tool_call_id")) {
		t.Fatalf("template JSON must omit IDs: %s", data)
	}
}

func TestUnsupportedFeaturesFail(t *testing.T) {
	cases := []string{
		`{"messages":[{"role":"user","content":[{"type":"file","file":{}}]}]}`,
		`{"messages":[{"role":"system","content":[{"type":"image_url","image_url":"a"}]}]}`,
		`{"messages":[{"role":"tool","tool_call_id":[]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x","detail":"low"}}]}]}`,
		`{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"v\":NaN}"}}]}]}`,
		`{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":[1.5]}}]}]}`,
		`{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","other":"lost"}}]}]}`,
		`{"tool_choice":"none"}`,
		`{"response_format":{"type":"json_object"}}`,
		`{"messages":[{"role":"user","content":"\ud800"}]}`,
		`{"messages":[{"role":"user","content":"x","content":"y"}]}`,
		`{"tools":[{"name":"x","parameters":{"default":1e400}}]}`,
	}
	n, _ := NewNormalizer(FamilyQwen)
	for _, c := range cases {
		t.Run(c, func(t *testing.T) {
			req, err := DecodeOpenAI([]byte(c))
			if err == nil {
				_, err = n.NormalizeOpenAI(req)
			}
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("expected explicit unsupported error, got %v", err)
			}
		})
	}
}

func TestOpenAIEmptyPartTypeRejectedButOmittedTypeAccepted(t *testing.T) {
	n, _ := NewNormalizer(FamilyQwen)
	for _, input := range []string{
		`{"messages":[{"role":"user","content":[{"type":"","text":"x"}]}]}`,
	} {
		req, err := DecodeOpenAI([]byte(input))
		if err == nil {
			_, err = n.NormalizeOpenAI(req)
		}
		if !errors.Is(err, ErrUnsupported) {
			t.Fatalf("expected empty type to be unsupported, got %v", err)
		}
	}
	req, err := DecodeOpenAI([]byte(`{"messages":[{"role":"user","content":[{"text":"x"}]}]}`))
	if err != nil {
		t.Fatalf("omitted type: %v", err)
	}
	result, err := n.NormalizeOpenAI(req)
	if err != nil || result.Messages[0].Content.Text != "x" {
		t.Fatalf("omitted type should be read as text: result %#v, error %v", result, err)
	}
}

func TestOrderedValuesAndUnicode(t *testing.T) {
	for _, input := range []string{
		`{"z":1,"a":{"y":2,"b":3}}`,
		`{"emoji":"\ud83d\ude00","literal":"\\u2028","line":"\u2028"}`,
	} {
		v, err := ParseValue([]byte(input))
		if err != nil {
			t.Fatal(err)
		}
		data, err := v.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		again, err := ParseValue(data)
		if err != nil {
			t.Fatalf("%s: %v", data, err)
		}
		reencoded, _ := again.MarshalJSON()
		if !bytes.Equal(data, reencoded) {
			t.Fatalf("unstable JSON: %s != %s", data, reencoded)
		}
		if strings.HasPrefix(input, `{"z":`) && string(data) != input {
			t.Fatalf("object order changed: %s", data)
		}
		if strings.Contains(input, `"literal"`) {
			literal, _ := again.Lookup("literal")
			if s, _ := literal.Text(); s != `\u2028` {
				t.Fatalf("literal escape changed: %q", s)
			}
		}
	}
	for _, input := range []string{`"\ud800"`, `"\udc00"`, `{"x":1,"x":2}`, `"bad` + string([]byte{0xff}) + `"`} {
		if _, err := ParseValue([]byte(input)); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("%q: expected unsupported, got %v", input, err)
		}
	}
}

func TestConcurrentToolOrder(t *testing.T) {
	n, _ := NewNormalizer(FamilyQwen)
	first, err := DecodeOpenAI([]byte(`{"tools":[{"name":"b"},{"name":"a"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	reversed, err := DecodeOpenAI([]byte(`{"tools":[{"name":"a"},{"name":"b"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := n.NormalizeOpenAI(first); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 128)
	for i := 0; i < cap(errs); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := n.NormalizeOpenAI(reversed)
			if err != nil {
				errs <- err
				return
			}
			name, _ := get(result.Tools[0], "name").Text()
			if name != "b" {
				errs <- fmt.Errorf("tool order: first %q", name)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
