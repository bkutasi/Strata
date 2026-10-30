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

type anthCase struct {
	Name         string          `json:"-"`
	Family       Family          `json:"family"`
	Request      json.RawMessage `json:"request"`
	ThinkUnasked bool            `json:"think_unasked"`
	Require      bool            `json:"require,omitempty"`
	API          string          `json:"api,omitempty"`
}

func anth(name, request string) anthCase {
	return anthCase{Name: name, Family: FamilyQwen, Request: json.RawMessage(request), ThinkUnasked: true}
}

func TestAnthropicDifferential(t *testing.T) {
	python := filepath.Join("..", "..", ".venv", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Skip("project .venv Python unavailable")
	}
	cases := []anthCase{
		anth("empty request", `{}`),
		{Name: "empty required", Family: FamilyQwen, Request: json.RawMessage(`{"messages":[]}`), ThinkUnasked: true, Require: true},
		anth("null messages", `{"messages":null}`),
		anth("basic string", `{"system":"be brief","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"answer"},{"role":"system","content":"late"}]}`),
		anth("empty system", `{"system":"","messages":[{"role":"user","content":""}]}`),
		anth("empty system array", `{"system":[],"messages":[{"role":"user","content":null}]}`),
		anth("system blocks", `{"system":[{"type":"text","text":"α"},{"text":"β"},{"type":"input_text","text":"γ"},{"type":"refusal","refusal":"no"}],"messages":[{"role":"user","content":"go"}]}`),
		anth("billing header", `{"system":"x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e; end"}`),
		anth("billing unicode", `{"system":"x-anthropic-billing-header: αα cch=☃x; cc_version=1.2.3.é.x; rest"}`),
		anth("billing start only", `{"system":"hello x-anthropic-billing-header: cch=abc; cc_version=1.2.3.4;"}`),
		anth("billing absent and bounded", `{"system":"x-anthropic-billing-header: cch=12345678901234567; cc_version=1.2.3;"}`),
		anth("text and empty blocks", `{"messages":[{"role":"user","content":[{"type":"text","text":"one"},{"type":"text"},{"type":"refusal","refusal":"no"},{"type":"text","text":"two"}]},{"role":"assistant","content":[]},{"role":"assistant","content":[{"type":"text","text":""}]}]}`),
		anth("thinking blocks", `{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"first"},{"type":"text","text":"ok"},{"type":"thinking","thinking":"second"}]},{"role":"assistant","content":[{"type":"thinking","thinking":""}]},{"role":"assistant","content":[{"type":"thinking"}]}]}`),
		anth("tool calls and result order", `{"messages":[{"role":"assistant","content":[{"type":"text","text":"run"},{"type":"tool_use","id":"a","name":"f","input":{"z":1,"a":{"b":2}}},{"type":"tool_use","id":"b","name":"g","input":{"q":"x"}}]},{"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"b","content":"B"},{"type":"text","text":"after"},{"type":"tool_result","tool_use_id":"a","content":[{"type":"text","text":"A"}]}]}]}`),
		anth("missing tool IDs and arguments", `{"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"a"},{"type":"tool_use","id":"b","name":"b","input":"{cut"},{"type":"tool_use","id":"c","name":"c","input":[1,{"x":true}]},{"type":"tool_use","id":"d","name":"d","input":"{\"z\":1,\"a\":2}"}]},{"role":"user","content":[{"type":"tool_result","content":null}]}]}`),
		anth("user images fast path", `{"messages":[{"role":"user","content":[{"type":"text"},{"type":"text","text":""},{"text":"see "},{"type":"input_text","text":"two "},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"AA=="}},{"type":"image","source":{"type":"url","url":"https://x"}},{"type":"image_url","image_url":{"url":"http://y"}},{"type":"refusal","refusal":"end"}]}]}`),
		anth("image defaults and assistant images", `{"messages":[{"role":"assistant","content":[{"type":"image","source":{"type":"base64","data":"a"}},{"type":"image","source":{"path":"a.png"}},{"type":"image_url"},{"type":"text","text":"caption"}]}]}`),
		anth("tool images and sibling images", `{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"i","name":"read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"i","content":[{"type":"text","text":"pic"},{"type":"image","source":{"type":"url","url":"x"}}]},{"type":"image","source":{"type":"base64","data":"AA=="}},{"type":"text","text":"outside"}]}]}`),
		anth("only tool results", `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"A"},{"type":"tool_result","tool_use_id":"b","content":[{"text":"B"},{"type":"input_text","text":"!"}]}]}]}`),
		anth("schema order and defaults", `{"tools":[{"name":"one","input_schema":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number","default":1.5}},"required":["z","a"]}},{"name":"two","description":"desc","input_schema":null},{"name":"three","description":null}]}`),
		anth("cache first", `{"tools":[{"name":"anth_b","input_schema":{"z":1,"a":2}},{"name":"anth_a","input_schema":{"a":1,"z":2}}]}`),
		anth("cache reversed metadata", `{"tools":[{"name":"anth_a","description":"new a"},{"name":"anth_b","description":"new b"}]}`),
		anth("duplicates first", `{"tools":[{"name":"dup","description":"one"},{"name":"dup","description":"two"}]}`),
		anth("duplicates reversed", `{"tools":[{"name":"dup","description":"two"},{"name":"dup","description":"one"}]}`),
		{Name: "cross API seed", Family: FamilyQwen, API: "openai", Request: json.RawMessage(`{"tools":[{"name":"cross_b"},{"name":"cross_a"}]}`), ThinkUnasked: true},
		anth("cross API Anthropic", `{"tools":[{"name":"cross_a","description":"a"},{"name":"cross_b","description":"b"}]}`),
		anth("thinking enabled", `{"thinking":{"type":"enabled","budget_tokens":500}}`),
		anth("budget 2047", `{"thinking":{"type":"enabled","budget_tokens":2047}}`),
		anth("budget 2048", `{"thinking":{"type":"enabled","budget_tokens":2048}}`),
		anth("budget 8191", `{"thinking":{"type":"enabled","budget_tokens":8191}}`),
		anth("budget 8192", `{"thinking":{"type":"enabled","budget_tokens":8192}}`),
		anth("budget float truncates", `{"thinking":{"type":"enabled","budget_tokens":2047.9}}`),
		anth("budget signed string", `{"thinking":{"type":"enabled","budget_tokens":" +8_192 "}}`),
		anth("budget invalid string", `{"thinking":{"type":"enabled","budget_tokens":"xyz"}}`),
		anth("budget invalid underscore", `{"thinking":{"type":"enabled","budget_tokens":"_8192"}}`),
		anth("budget boolean", `{"thinking":{"type":"enabled","budget_tokens":true}}`),
		anth("budget negative", `{"thinking":{"type":"enabled","budget_tokens":-1}}`),
		anth("budget zero", `{"thinking":{"type":"enabled","budget_tokens":0}}`),
		anth("budget explicit zero wins", `{"thinking":{"type":"enabled","budget_tokens":100},"reasoning_budget_tokens":0}`),
		anth("budget null falls back", `{"thinking":{"type":"enabled","budget_tokens":100},"reasoning_budget_tokens":null}`),
		anth("effort wins level not budget", `{"thinking":{"type":"enabled","budget_tokens":100},"output_config":{"effort":"HIGH"}}`),
		anth("disabled wins effort", `{"thinking":{"type":"disabled","budget_tokens":100},"output_config":{"effort":"bad"}}`),
		anth("false effort falls through", `{"thinking":{"type":"enabled","budget_tokens":8192},"output_config":{"effort":false}}`),
		anth("adaptive", `{"thinking":{"type":"adaptive"}}`),
		anth("unknown effort", `{"output_config":{"effort":"impossible"}}`),
		{Name: "on request default", Family: FamilyQwen, Request: json.RawMessage(`{}`), ThinkUnasked: false},
		{Name: "on request explicit budget", Family: FamilyQwen, Request: json.RawMessage(`{"reasoning_budget_tokens":30}`), ThinkUnasked: false},
		{Name: "on request zero budget", Family: FamilyQwen, Request: json.RawMessage(`{"reasoning_budget_tokens":0}`), ThinkUnasked: false},
		{Name: "on request empty thinking", Family: FamilyQwen, Request: json.RawMessage(`{"thinking":{}}`), ThinkUnasked: false},
		{Name: "deepseek budget", Family: FamilyDeepSeek, Request: json.RawMessage(`{"thinking":{"type":"enabled","budget_tokens":8192}}`), ThinkUnasked: true},
		{Name: "deepseek max", Family: FamilyDeepSeek, Request: json.RawMessage(`{"output_config":{"effort":"xhigh"}}`), ThinkUnasked: true},
		{Name: "deepseek numeric", Family: FamilyDeepSeek, Request: json.RawMessage(`{"output_config":{"effort":37}}`), ThinkUnasked: true},
		{Name: "deepseek invalid numeric", Family: FamilyDeepSeek, Request: json.RawMessage(`{"output_config":{"effort":101}}`), ThinkUnasked: true},
		anth("bad messages", `{"messages":"not json"}`),
		anth("nonobject messages", `{"messages":[{"role":"user","content":"x"},7]}`),
		anth("bad tool list", `{"tools":"auto"}`),
		anth("tool missing name", `{"tools":[{"input_schema":{}}]}`),
		anth("bad tool schema", `{"tools":[{"name":"f","input_schema":[]}]}`),
		anth("bad tool schema bool", `{"tools":[{"name":"f","input_schema":true}]}`),
	}
	encodedMessages := `[{"role":"user","content":"hello"},{"role":"assistant","content":[{"type":"text","text":"ok"}]}]`
	cases = append(cases, anth("double encoded messages", `{"messages":`+encodeJSON(t, encodedMessages)+`}`))
	cases = append(cases, anth("double encoded tools", `{"tools":`+encodeJSON(t, `[{"name":"encoded","input_schema":{"z":1,"a":2}}]`)+`}`))
	cases = append(cases, anth("oversized budget string",
		`{"thinking":{"type":"enabled","budget_tokens":"`+strings.Repeat("0", 4301)+`"}}`))

	var stdin bytes.Buffer
	for _, c := range cases {
		if err := json.NewEncoder(&stdin).Encode(c); err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
	}
	cmd := exec.Command(python, "anth_oracle.py")
	cmd.Stdin = &stdin
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("reference failed: %v: %s", err, stderr.String())
	}
	norms := make(map[Family]*Normalizer)
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for _, c := range cases {
		if !scanner.Scan() {
			t.Fatalf("%s: reference stopped: %v; stderr: %s", c.Name, scanner.Err(), stderr.String())
		}
		var ref struct {
			Result json.RawMessage `json:"result"`
			Budget json.RawMessage `json:"budget"`
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
		var got any
		var budget Value
		var gotErr error
		if c.API == "openai" {
			var req OpenAIRequest
			req, gotErr = DecodeOpenAI(c.Request)
			if gotErr == nil {
				var result Result
				result, gotErr = n.NormalizeOpenAI(req)
				if gotErr == nil && c.Require {
					gotErr = RequireMessages(result.Messages)
				}
				got = result
			}
		} else {
			var req AnthropicRequest
			req, gotErr = DecodeAnthropic(c.Request)
			if gotErr == nil {
				var result AnthropicResult
				result, gotErr = n.NormalizeAnthropic(req, c.ThinkUnasked)
				if gotErr == nil && c.Require {
					gotErr = RequireAnthropicMessages(result.Messages)
				}
				got, budget = result, result.ReasoningBudgetTokens
			}
		}
		if ref.Error != "" || gotErr != nil {
			// Compare success/error only, not user-facing prose.
			if (ref.Error == "") != (gotErr == nil) {
				t.Errorf("%s: reference error %q; Go error %v", c.Name, ref.Error, gotErr)
			}
			continue
		}
		gotJSON, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("%s: marshal result: %v", c.Name, err)
		}
		assertOrderedJSON(t, c.Name, gotJSON, ref.Result)
		budgetJSON, err := budget.MarshalJSON()
		if err != nil {
			t.Fatalf("%s: marshal budget: %v", c.Name, err)
		}
		assertOrderedJSON(t, c.Name+" budget", budgetJSON, ref.Budget)
	}
	if scanner.Scan() {
		t.Fatalf("reference produced extra output: %s", scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
}

func assertOrderedJSON(t *testing.T, name string, gotJSON, wantJSON []byte) {
	t.Helper()
	got, err := ParseValue(gotJSON)
	if err != nil {
		t.Fatalf("%s: parse result: %v", name, err)
	}
	want, err := ParseValue(wantJSON)
	if err != nil {
		t.Fatalf("%s: parse reference: %v", name, err)
	}
	gotCanonical, err := got.MarshalJSON()
	if err != nil {
		t.Fatalf("%s: serialize result: %v", name, err)
	}
	wantCanonical, err := want.MarshalJSON()
	if err != nil {
		t.Fatalf("%s: serialize reference: %v", name, err)
	}
	if !bytes.Equal(gotCanonical, wantCanonical) {
		t.Errorf("%s:\nGo:        %s\nreference: %s", name, gotCanonical, wantCanonical)
	}
}

func TestAnthropicIDsRemainInReceivedOrder(t *testing.T) {
	req, err := DecodeAnthropic([]byte(`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}},{"type":"tool_use","id":"b","name":"g","input":{}}]},{"role":"user","content":[{"type":"text","text":"alongside"},{"type":"tool_result","tool_use_id":"b","content":"B"},{"type":"tool_result","tool_use_id":"a","content":"A"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	n, _ := NewNormalizer(FamilyQwen)
	result, err := n.NormalizeAnthropic(req, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 4 || result.Messages[0].ToolCalls[0].ID != "a" ||
		result.Messages[0].ToolCalls[1].ID != "b" || result.Messages[1].ToolCallID != "b" ||
		result.Messages[1].Content.Text != "B" || result.Messages[2].ToolCallID != "a" ||
		result.Messages[2].Content.Text != "A" || result.Messages[3].Content.Text != "alongside" {
		t.Fatalf("IDs or received order lost: %#v", result.Messages)
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(`"id"`)) || bytes.Contains(data, []byte("tool_call_id")) ||
		bytes.Contains(data, []byte("tool_use_id")) {
		t.Fatalf("template JSON must omit IDs: %s", data)
	}
}

func TestAnthropicUnsupportedFeaturesFail(t *testing.T) {
	cases := []string{
		`{"model":"m"}`,
		`{"tool_choice":{"type":"none"}}`,
		`{"system":[{"type":"image","source":{"type":"url","url":"x"}}]}`,
		`{"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}`,
		`{"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","data":"x"}]}]}`,
		`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"x","signature":"sig"}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"image","source":{}},{"type":"tool_use","name":"f","input":{}}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":[],"content":"x"}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","is_error":true}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"input_text","text":"not read"}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"","text":"not read"},{"type":"image","source":{}}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"tool_result","content":[{"type":"","text":"not read"}]}]}]}`,
		`{"system":[{"type":"","text":"not read"}]}`,
		`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"x","input":{}}]}]}`,
		`{"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"f","input":[1.5]}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"x"}}]}]}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`,
		`{"tools":[{"name":"f","input_schema":{},"cache_control":{"type":"ephemeral"}}]}`,
		`{"thinking":{"type":"unknown"}}`,
		`{"thinking":{"type":"enabled","budget_tokens":"１２３"}}`,
		`{"output_config":{"effort":{"level":"high"}}}`,
		`{"reasoning_budget_tokens":true}`,
		`{"messages":[{"role":"user","content":"\ud800"}]}`,
		`{"messages":[{"role":"user","content":"x","content":"y"}]}`,
		`{"tools":[{"name":"x","input_schema":{"default":1e400}}]}`,
	}
	n, _ := NewNormalizer(FamilyQwen)
	for _, input := range cases {
		t.Run(input, func(t *testing.T) {
			req, err := DecodeAnthropic([]byte(input))
			if err == nil {
				_, err = n.NormalizeAnthropic(req, true)
			}
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("expected unsupported error, got %v", err)
			}
		})
	}
}

func TestAnthropicBudgetDoesNotMutateRequest(t *testing.T) {
	req, err := DecodeAnthropic([]byte(`{"thinking":{"type":"enabled","budget_tokens":2048}}`))
	if err != nil {
		t.Fatal(err)
	}
	n, _ := NewNormalizer(FamilyQwen)
	result, err := n.NormalizeAnthropic(req, false)
	if err != nil {
		t.Fatal(err)
	}
	if req.ReasoningBudgetTokens.Kind() != Null {
		t.Fatal("request was mutated")
	}
	text, ok := result.ReasoningBudgetTokens.Text()
	if !ok || text != "2048" {
		t.Fatalf("effective budget = %q, valid %t", text, ok)
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "reasoning_budget_tokens") {
		t.Fatalf("template JSON includes runtime budget: %s", data)
	}
}

func TestAnthropicConcurrentSharedToolOrder(t *testing.T) {
	n, _ := NewNormalizer(FamilyQwen)
	first, err := DecodeOpenAI([]byte(`{"tools":[{"name":"shared_b"},{"name":"shared_a"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := n.NormalizeOpenAI(first); err != nil {
		t.Fatal(err)
	}
	req, err := DecodeAnthropic([]byte(`{"tools":[{"name":"shared_a"},{"name":"shared_b"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for i := 0; i < cap(errs); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := n.NormalizeAnthropic(req, true)
			if err != nil {
				errs <- err
				return
			}
			name, _ := get(result.Tools[0], "name").Text()
			if name != "shared_b" {
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
