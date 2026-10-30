package template

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func source(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", path))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestGolden(t *testing.T) {
	python := filepath.Join("..", "..", ".venv", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		t.Fatalf("golden tests need project .venv with Jinja2: %v", err)
	}
	cmd := exec.Command(python, "oracle.py")
	data, err := cmd.Output()
	if err != nil {
		t.Fatalf("golden input oracle: %v", err)
	}
	qwen, err := New(source(t, "serve/chat_template.jinja"))
	if err != nil {
		t.Fatal(err)
	}
	deep, err := New(source(t, "serve/fixtures/deepseek41/chat_template.jinja"))
	if err != nil {
		t.Fatal(err)
	}
	var count int
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 4096), 4<<20)
	for sc.Scan() {
		var c struct {
			Name, Family            string
			Messages, Tools, Kwargs json.RawMessage
			Generate                bool
			Expected                string
		}
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			t.Fatal(err)
		}
		count++
		t.Run(c.Family+"/"+c.Name, func(t *testing.T) {
			r := qwen
			if c.Family == "deepseek" {
				r = deep
			}
			got, err := r.Render(context.Background(), Input{
				Messages: c.Messages, Tools: c.Tools, Kwargs: c.Kwargs,
				AddGenerationPrompt: c.Generate,
			})
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Expected {
				t.Errorf("render differs from golden:\ngot  %q\nwant %q", got, c.Expected)
			}
		})
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if count != 16 {
		t.Fatalf("expected 16 fixture cases, got %d", count)
	}
}

func TestCompatibility(t *testing.T) {
	tests := []struct {
		name, source, messages, tools, kwargs, want string
		keep                                        bool
	}{
		{"ordered unicode numbers", `{{ tools | tojson }}`, `[]`, `[{"z":9007199254740993,"a":{"f":1000000000000000.0,"s":0.00001,"u":"é<&\u2028\\u2028"}}]`, `{}`, `[{"z": 9007199254740993, "a": {"f": 1000000000000000.0, "s": 1e-05, "u": "é<& \\u2028"}}]`, false},
		{"integer and float types", `{{ tools[0] is integer }} {{ tools[1] is float }} {{ tools[0] + 1 }}`, `[]`, `[9223372036854775809,1.0]`, `{}`, `True True 9223372036854775810`, false},
		{"empty assistant removed", `{{ messages | length }}`, `[{"role":"assistant","content":" \n"},{"role":"user","content":"hi"}]`, `null`, `{}`, `1`, false},
		{"keep empty", `{{ messages | length }}`, `[{"role":"assistant","content":""},{"role":"user","content":"hi"}]`, `null`, `{}`, `2`, true},
		{"final empty kept", `{{ messages | length }}`, `[{"role":"assistant","content":""}]`, `null`, `{}`, `1`, false},
		{"tool calls kept", `{{ messages | length }}`, `[{"role":"assistant","content":"","tool_calls":[{"name":"x"}]},{"role":"user","content":"hi"}]`, `null`, `{}`, `2`, false},
		{"image kept", `{{ messages | length }}`, `[{"role":"assistant","content":[{"type":"image"}]},{"role":"user","content":"hi"}]`, `null`, `{}`, `2`, false},
		{"whitespace", "a\n{% if true %}\n  {{ 'x' }}\n{% endif %}\nb", `[]`, `null`, `{}`, "a\n  x\nb", false},
		{"untrusted source stays data", `{{ messages[0].content }}`, `[{"role":"user","content":"{{ 7 * 7 }}"}]`, `null`, `{}`, `{{ 7 * 7 }}`, false},
	}
	for _, c := range tests {
		t.Run(c.name, func(t *testing.T) {
			r, err := New(c.source)
			if err != nil {
				t.Fatal(err)
			}
			got, err := r.Render(context.Background(), Input{Messages: []byte(c.messages), Tools: []byte(c.tools), Kwargs: []byte(c.kwargs), KeepEmptyTurns: c.keep})
			if err != nil || got != c.want {
				t.Fatalf("got %q, error %v; want %q", got, err, c.want)
			}
		})
	}
}

func TestErrors(t *testing.T) {
	r, err := New(`{{ raise_exception('refused') }}`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = r.Render(context.Background(), Input{Messages: []byte(`[]`)})
	var requestErr *RequestError
	if !errors.As(err, &requestErr) {
		t.Fatalf("expected RequestError, got %v", err)
	}
	_, err = r.Render(context.Background(), Input{Messages: []byte(`[]`), Kwargs: []byte(`{"messages":[]}`)})
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("expected unsupported reserved argument, got %v", err)
	}
	r, err = New(`{{ tools | tojson(indent=2) }}`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = r.Render(context.Background(), Input{Messages: []byte(`[]`)})
	if err == nil || !strings.Contains(err.Error(), "tojson") {
		t.Fatalf("expected unsupported JSON options, got %v", err)
	}
}
