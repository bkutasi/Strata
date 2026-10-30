package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"strata/requestnorm"
)

func TestSplitJoinedConfigArgs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	data := `{"exe":"e","tokenizer":"t","model_name":"m","args":["--ple-io mmap","--prefill=auto","--pack","--not-an-option=literal","model file.gguf","--flag=","--empty "]}`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := readStandaloneConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--ple-io", "mmap", "--prefill", "auto", "--pack", "--not-an-option=literal", "model file.gguf", "--flag=", "--empty "}
	if !reflect.DeepEqual(cfg.Args, want) {
		t.Fatalf("args = %#v, want %#v", cfg.Args, want)
	}
}

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

func TestStandaloneFitMaxTokensConfig(t *testing.T) {
	for _, tc := range []struct {
		value, want string
	}{
		{``, ""},
		{`true`, "true"},
		{`false`, "false"},
		{`null`, "null"},
		{`"false"`, `"false"`},
		{`0`, "0"},
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		data := `{"exe":"e","tokenizer":"t","model_name":"m"`
		if tc.value != "" {
			data += `,"fit_max_tokens":` + tc.value
		}
		data += `}`
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		cfg, err := readStandaloneConfig(path)
		if err != nil || string(cfg.FitMaxTokens) != tc.want {
			t.Errorf("%s: config %s, error %v", tc.value, cfg.FitMaxTokens, err)
		}
	}
}

func TestStandaloneAliasesConfig(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  string
		ok    bool
	}{
		{`[" a ","model","b","a",""]`, `["a","b"]`, true},
		{`" a, b,a,model,"`, `["a","b"]`, true},
		{`17`, "", false},
		{`["a",null]`, "", false},
		{`["a",1]`, "", false},
	} {
		path := filepath.Join(t.TempDir(), "config.json")
		data := `{"exe":"e","tokenizer":"t","model_name":"model","aliases":` + tc.value + `}`
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		cfg, err := readStandaloneConfig(path)
		if (err == nil) != tc.ok {
			t.Fatalf("%s: error %v", tc.value, err)
		}
		if !tc.ok {
			continue
		}
		aliases, err := cfg.aliasesConfig()
		encoded, _ := json.Marshal(aliases)
		if err != nil || string(encoded) != tc.want {
			t.Errorf("%s: aliases %s, error %v", tc.value, encoded, err)
		}
	}
}

func TestStandaloneConfigRejectsUnknownAndTrailingData(t *testing.T) {
	for _, data := range []string{
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

func TestStandaloneVisionConfig(t *testing.T) {
	for _, tc := range []struct {
		name   string
		field  string
		family string
		want   []string
		ok     bool
	}{
		{"cpu defaults", `"vision":{"exe":"reader","mmproj":"mm/model.gguf","model":"weights/model.gguf"}`, "qwen",
			[]string{filepath.Join(mustGetwd(t), "reader"), "--mmproj", filepath.Join(mustGetwd(t), "mm/model.gguf"), "--model", filepath.Join(mustGetwd(t), "weights/model.gguf"), "--max-tokens", "300"}, true},
		{"gpu defaults", `"vision":{"exe":"reader","mmproj":"mm.gguf","model":"model.gguf","gpu":true}`, "qwen",
			[]string{filepath.Join(mustGetwd(t), "reader"), "--mmproj", filepath.Join(mustGetwd(t), "mm.gguf"), "--model", filepath.Join(mustGetwd(t), "model.gguf"), "--gpu", "--max-tokens", "1024"}, true},
		{"zero limit omitted", `"vision":{"exe":"reader","mmproj":"mm.gguf","model":"model.gguf","max_tokens":0}`, "qwen",
			[]string{filepath.Join(mustGetwd(t), "reader"), "--mmproj", filepath.Join(mustGetwd(t), "mm.gguf"), "--model", filepath.Join(mustGetwd(t), "model.gguf")}, true},
		{"unsupported field", `"vision":{"exe":"reader","mmproj":"m","model":"w","lazy":true}`, "qwen", nil, false},
		{"deepseek rejected", `"vision":{"exe":"reader","mmproj":"m","model":"w"}`, "deepseek41", nil, false},
		{"missing required", `"vision":{"exe":"reader","mmproj":"m"}`, "qwen", nil, false},
		{"bad threads", `"vision":{"exe":"reader","mmproj":"m","model":"w","threads":-1}`, "qwen", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			data := `{"exe":"e","tokenizer":"t","model_name":"m",` + tc.field + `}`
			if err := os.WriteFile(path, []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
			cfg, err := readStandaloneConfig(path)
			if err != nil {
				t.Fatal(err)
			}
			family := requestnorm.FamilyQwen
			if tc.family == "deepseek41" {
				family = requestnorm.FamilyDeepSeek
			}
			_, args, err := cfg.visionConfig(family)
			if (err == nil) != tc.ok {
				t.Fatalf("visionConfig error = %v, want ok=%v", err, tc.ok)
			}
			if tc.ok && !reflect.DeepEqual(args, tc.want) {
				t.Fatalf("vision args = %#v, want %#v", args, tc.want)
			}
		})
	}
}

func TestStandaloneVisionPathsUseConfigCwd(t *testing.T) {
	dir := t.TempDir()
	cfg := standaloneConfig{Cwd: dir, Vision: json.RawMessage(`{"exe":"reader","mmproj":"mm.gguf","model":"model.gguf"}`)}
	visionCfg, args, err := cfg.visionConfig(requestnorm.FamilyQwen)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{filepath.Join(dir, "reader"), "--mmproj", filepath.Join(dir, "mm.gguf"), "--model", filepath.Join(dir, "model.gguf"), "--max-tokens", "300"}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("vision args = %#v, want %#v", args, want)
	}
	if visionCfg.Exe != want[0] || visionCfg.MMProj != want[2] || visionCfg.Model != want[4] {
		t.Fatalf("resolved vision config = %+v", visionCfg)
	}
}

func mustGetwd(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return dir
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
