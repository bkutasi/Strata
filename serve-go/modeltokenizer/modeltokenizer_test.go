package modeltokenizer

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

func syntheticPack(t *testing.T, pre, model string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "tokenizer")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	vocab := make(map[string]int)
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = rune(256 + n)
			n++
		}
		vocab[string(r)] = b
	}
	vocab["12"], vocab["ab"], vocab["<x>"] = 256, 257, 258
	types := make([]int, len(vocab))
	types[258] = 3
	writeJSON := func(name string, value any) {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON("tokenizer.json", map[string]string{"pre": pre, "model": model})
	writeJSON("vocab.json", vocab)
	writeJSON("token_type.json", types)
	if err := os.WriteFile(filepath.Join(dir, "merges.txt"), []byte("1 2\na b"), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestIntegration(t *testing.T) {
	for _, tc := range []struct {
		pre  string
		want []int
	}{
		{"qwen35", []int{'1', '2', 257}},
		{"deepseek3", []int{256, 257}},
		{"deepseek-v3", []int{256, 257}},
	} {
		t.Run(tc.pre, func(t *testing.T) {
			dir := syntheticPack(t, tc.pre, "gpt2")
			tok, err := Load(filepath.Dir(dir))
			if err != nil {
				t.Fatal(err)
			}
			got, err := tok.Encode("12ab", false, nil)
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Encode: %v, %v; want %v", got, err, tc.want)
			}
			got, err = tok.Encode("é<x><x>", true, []Span{{Start: 2, End: 5}})
			want := []int{195, 169, '<', 'x', '>', 258}
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("protected byte span: %v, %v; want %v", got, err, want)
			}
			decoded, err := tok.Decode(got)
			if err != nil || decoded != "é<x><x>" {
				t.Fatalf("Decode: %q, %v", decoded, err)
			}
			if _, err = tok.EncodeWithBudget("12", false, nil, 1); err == nil {
				t.Fatal("budget not enforced")
			}
		})
	}
}

func TestRejectUnknownMetadata(t *testing.T) {
	for _, tc := range [][2]string{{"qwen3", "gpt2"}, {"qwen35", "other"}} {
		if _, err := Load(syntheticPack(t, tc[0], tc[1])); err == nil {
			t.Fatalf("accepted pre=%q model=%q", tc[0], tc[1])
		}
	}
	if _, err := Load(syntheticPack(t, "deepseek3", "")); err != nil {
		t.Fatal(err)
	}
}

func TestLegacyMetadata(t *testing.T) {
	for _, metadata := range []string{"", "{", "{}", `{"pre":""}`, `{"pre":null}`, `{"pre":false}`, `{"pre":0}`, `{"pre":0.0}`, `{"pre":-0.0}`, `{"pre":0e10}`, `{"pre":[ ]}`, `{"pre":{ }}`} {
		t.Run(metadata, func(t *testing.T) {
			dir := syntheticPack(t, "qwen35", "gpt2")
			path := filepath.Join(dir, "tokenizer.json")
			var err error
			if metadata == "" {
				err = os.Remove(path)
			} else {
				err = os.WriteFile(path, []byte(metadata), 0o600)
			}
			if err != nil {
				t.Fatal(err)
			}
			tok, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}
			got, err := tok.Encode("12", false, nil)
			if err != nil || !reflect.DeepEqual(got, []int{'1', '2'}) {
				t.Fatalf("legacy qwen split: %v, %v", got, err)
			}
		})
	}
}

func TestRejectTruthyNonFamily(t *testing.T) {
	for _, value := range []string{`"unknown"`, `" "`, `true`, `1`, `-0.1`, `[0]`, `{"x":0}`} {
		dir := syntheticPack(t, "qwen35", "gpt2")
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"pre":`+value+`}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(dir); err == nil {
			t.Errorf("accepted pre %s", value)
		}
	}
}

func TestSpecialIDs(t *testing.T) {
	for _, ids := range []map[string]int{
		{"tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.eos_token_id": 258, "tokenizer.ggml.padding_token_id": 2},
		{"bos": 0, "eos": 258, "pad": 2},
	} {
		dir := syntheticPack(t, "qwen35", "gpt2")
		data, err := json.Marshal(map[string]any{"pre": "qwen35", "special_ids": ids})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0o600); err != nil {
			t.Fatal(err)
		}
		tok, err := Load(dir)
		if err != nil {
			t.Fatal(err)
		}
		got := tok.SpecialIDs()
		if !reflect.DeepEqual(got, ids) {
			t.Fatalf("got %v, want %v", got, ids)
		}
		for key := range got {
			got[key] = -1
		}
		if !reflect.DeepEqual(tok.SpecialIDs(), ids) {
			t.Fatal("returned map aliases tokenizer metadata")
		}
	}
	for _, id := range []string{"-1", "259", "null", `"1"`, "1.5", "true"} {
		dir := syntheticPack(t, "qwen35", "gpt2")
		if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"special_ids":{"eos":`+id+`}}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(dir); err == nil {
			t.Fatalf("accepted invalid ID %s", id)
		}
	}
}

func TestBudget(t *testing.T) {
	tok, err := Load(syntheticPack(t, "qwen35", "gpt2"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		text string
		max  int
		ok   bool
	}{
		{"", 0, true}, {"é", 2, true}, {"é", 1, false},
		{"<x>", 2, false}, {"", -1, false},
	} {
		_, err := tok.EncodeWithBudget(tc.text, true, nil, tc.max)
		if (err == nil) != tc.ok {
			t.Errorf("%q budget %d: %v", tc.text, tc.max, err)
		}
	}
}

// Set STRATA_TOKENIZER_PACK and STRATA_REFERENCE_PYTHON explicitly to opt in.
// This may consume substantial memory and is not part of lightweight checks.
func TestOptionalPackDifferential(t *testing.T) {
	pack, python := os.Getenv("STRATA_TOKENIZER_PACK"), os.Getenv("STRATA_REFERENCE_PYTHON")
	if pack == "" || python == "" {
		t.Skip("set STRATA_TOKENIZER_PACK and STRATA_REFERENCE_PYTHON to enable")
	}
	tok, err := Load(pack)
	if err != nil {
		t.Fatal(err)
	}
	dir := pack
	if _, err := os.Stat(filepath.Join(pack, "tokenizer")); err == nil {
		dir = filepath.Join(pack, "tokenizer")
	}
	texts := []string{"Hello 1234!", "é中文 😀", "a\n b\tc", "<|im_end|> plain"}
	input, err := json.Marshal(texts)
	if err != nil {
		t.Fatal(err)
	}
	const script = `import json, pathlib, sys
sys.path.insert(0, "../../tools")
from strata_tokenizer import Tokenizer
p = pathlib.Path(sys.argv[1])
cfg = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for token, i in vocab.items(): tokens[i] = token
types = json.loads((p / "token_type.json").read_text(encoding="utf-8"))
merges = (p / "merges.txt").read_text(encoding="utf-8").splitlines()
tk = Tokenizer(tokens, merges, types, cfg["pre"])
print(json.dumps([tk.encode(s, parse_special=True) for s in json.loads(sys.stdin.read())]))
`
	cmd := exec.Command(python, "-c", script, dir)
	cmd.Stdin = bytes.NewReader(input)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python reference: %v: %s", err, out)
	}
	var reference [][]int
	if err := json.Unmarshal(out, &reference); err != nil {
		t.Fatal(err)
	}
	for i, text := range texts {
		got, err := tok.Encode(text, true, nil)
		if err != nil || !reflect.DeepEqual(got, reference[i]) {
			t.Fatalf("%q: %v, %v; want %v", text, got, err, reference[i])
		}
	}
}
