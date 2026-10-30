package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStandaloneConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := `{"exe":"bin/strata","cwd":"` + filepath.ToSlash(dir) + `","args":["--native","model"],` +
		`"tokenizer":"pack/tokenizer","model_name":"model","host":"127.0.0.1","gpu":0,` +
		`"api_key":"key","env":{"STRATA_TEST_SETTING":"value"},"log":"engine.log"}`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := readStandaloneConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	env, err := cfg.childEnv()
	if err != nil || cfg.Exe != filepath.Join(dir, "bin", "strata") || cfg.Model != "model" ||
		len(cfg.Args) != 2 || cfg.Args[1] != "model" ||
		!strings.Contains(strings.Join(env, "\n"), "CUDA_VISIBLE_DEVICES=0") ||
		!strings.Contains(strings.Join(env, "\n"), "STRATA_TEST_SETTING=value") {
		t.Fatalf("config or environment incorrect: %+v, %v", cfg, err)
	}
}

func TestStandaloneConfigRejectsUnknownAndTrailingData(t *testing.T) {
	for _, data := range []string{
		`{"exe":"e","tokenizer":"t","model_name":"m","vision":{}}`,
		`{"exe":"e","tokenizer":"t","model_name":"m"} {}`,
		`{"exe":"e","tokenizer":"t"}`,
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := readStandaloneConfig(path); err == nil {
			t.Fatalf("accepted invalid config %s", data)
		}
	}
}

func TestOptionalStandaloneConfig(t *testing.T) {
	path := os.Getenv("STRATA_TEST_CONFIG_PATH")
	if path == "" {
		t.Skip("set STRATA_TEST_CONFIG_PATH to check an existing config")
	}
	cfg, err := readStandaloneConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cfg.childEnv(); err != nil {
		t.Fatal(err)
	}
}

func TestStandaloneThinkingConfig(t *testing.T) {
	for _, tc := range []struct {
		fields string
		ok     bool
	}{
		{`"reasoning_budget_tokens":20,"answer_room":false`, true},
		{`"reasoning_budget_tokens":0.0,"answer_room":null`, true},
		{`"reasoning_budget_tokens":-1,"answer_room":"false"`, true},
		{`"reasoning_budget_tokens":true`, false},
		{`"reasoning_budget_tokens":1.5`, false},
		{`"reasoning_budget_tokens":"20"`, false},
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		data := `{"exe":"e","tokenizer":"t","model_name":"m",` + tc.fields + `}`
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := readStandaloneConfig(path)
		if (err == nil) != tc.ok {
			t.Errorf("%s: error %v, want ok=%v", tc.fields, err, tc.ok)
		}
	}
}

func TestStandaloneRepeatStopConfig(t *testing.T) {
	for _, tc := range []struct {
		field string
		ok    bool
	}{
		{``, true},
		{`,"repeat_stop_tokens":0`, true},
		{`,"repeat_stop_tokens":256`, true},
		{`,"repeat_stop_tokens":-1`, false},
		{`,"repeat_stop_tokens":true`, false},
		{`,"repeat_stop_tokens":null`, false},
		{`,"repeat_stop_tokens":1.0`, false},
		{`,"repeat_stop_tokens":"3"`, false},
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		data := `{"exe":"e","tokenizer":"t","model_name":"m"` + tc.field + `}`
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := readStandaloneConfig(path)
		if (err == nil) != tc.ok {
			t.Errorf("%s: error %v, want ok=%v", tc.field, err, tc.ok)
		}
	}
}

func TestStandaloneAnthropicThinkingConfig(t *testing.T) {
	for _, tc := range []struct {
		field string
		ok    bool
	}{
		{``, true},
		{`,"anthropic_thinking":"model"`, true},
		{`,"anthropic_thinking":"on_request"`, true},
		{`,"anthropic_thinking":null`, true},
		{`,"anthropic_thinking":"unknown"`, false},
		{`,"anthropic_thinking":true`, false},
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		data := `{"exe":"e","tokenizer":"t","model_name":"m"` + tc.field + `}`
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := readStandaloneConfig(path)
		if (err == nil) != tc.ok {
			t.Errorf("%s: error %v, want ok=%v", tc.field, err, tc.ok)
		}
	}
}

func TestStandaloneReasoningEffortConfig(t *testing.T) {
	for _, tc := range []struct {
		field string
		ok    bool
	}{
		{``, true}, {`,"reasoning_effort":null`, true}, {`,"reasoning_effort":""`, true},
		{`,"reasoning_effort":"none"`, true}, {`,"reasoning_effort":"low"`, true},
		{`,"reasoning_effort":"medium"`, true}, {`,"reasoning_effort":"high"`, true},
		{`,"reasoning_effort":"xhigh"`, false}, {`,"reasoning_effort":true`, false},
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		if err := os.WriteFile(path, []byte(`{"exe":"e","tokenizer":"t","model_name":"m"`+tc.field+`}`), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := readStandaloneConfig(path)
		if (err == nil) != tc.ok {
			t.Errorf("%s: error %v, want ok=%v", tc.field, err, tc.ok)
		}
	}
}
