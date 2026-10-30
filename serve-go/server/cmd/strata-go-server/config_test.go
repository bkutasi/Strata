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
