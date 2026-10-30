package template

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestOptionalPackTemplateDifferential(t *testing.T) {
	dir := os.Getenv("STRATA_TEMPLATE_PACK")
	if dir == "" {
		t.Skip("set STRATA_TEMPLATE_PACK to compare an installed template")
	}
	path := filepath.Join(dir, "chat_template.jinja")
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	renderer, err := New(string(source))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		Messages json.RawMessage `json:"messages"`
		Tools    json.RawMessage `json:"tools"`
		Kwargs   json.RawMessage `json:"kwargs"`
	}{
		{json.RawMessage(`[{"role":"user","content":"Hello, 世界"}]`), json.RawMessage(`null`), json.RawMessage(`{}`)},
		{json.RawMessage(`[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]`), json.RawMessage(`null`), json.RawMessage(`{"enable_thinking":false}`)},
		{json.RawMessage(`[{"role":"user","content":"Use lookup."}]`), json.RawMessage(`[{"name":"lookup","description":"Lookup a value","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}]`), json.RawMessage(`{"enable_thinking":false}`)},
	}
	input, err := json.Marshal(cases)
	if err != nil {
		t.Fatal(err)
	}
	script := `import json, sys
sys.path.insert(0, "../..")
from serve.frontend import ChatTemplate
template = ChatTemplate(sys.argv[1])
cases = json.load(sys.stdin)
json.dump([template.render(c["messages"], tools=c["tools"], **c["kwargs"]) for c in cases], sys.stdout, ensure_ascii=False)
`
	cmd := exec.Command("../../.venv/bin/python", "-c", script, path)
	cmd.Stdin = bytes.NewReader(input)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python renderer: %v: %s", err, output)
	}
	var expected []string
	if err := json.Unmarshal(output, &expected); err != nil {
		t.Fatal(err)
	}
	if len(expected) != len(cases) {
		t.Fatalf("got %d reference outputs for %d inputs", len(expected), len(cases))
	}
	for i, tc := range cases {
		got, err := renderer.Render(context.Background(), Input{
			Messages: tc.Messages, Tools: tc.Tools, Kwargs: tc.Kwargs, AddGenerationPrompt: true,
		})
		if err != nil || got != expected[i] {
			t.Errorf("case %d differs from Python: error %v\nGo: %q\nPython: %q", i, err, got, expected[i])
		}
	}
}
