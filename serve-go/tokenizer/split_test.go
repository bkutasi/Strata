package tokenizer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

const unicodeMaskSHA256 = "936112a46135affeed68629aba9b4ee89c4580c6e752714b7cc9f4310b09d1e8"

func TestUnicodeClasses(t *testing.T) {
	if unicodeBoundaries[0]>>6 != 0 {
		t.Fatal("first Unicode range does not start at zero")
	}
	for i := 1; i < len(unicodeBoundaries); i++ {
		if unicodeBoundaries[i]>>6 <= unicodeBoundaries[i-1]>>6 {
			t.Fatalf("Unicode boundaries out of order at %d", i)
		}
	}
	masks := make([]byte, utf8.MaxRune+1)
	for r := range masks {
		masks[r] = runeClass(rune(r))
	}
	sum := sha256.Sum256(masks)
	if got := hex.EncodeToString(sum[:]); got != unicodeMaskSHA256 {
		t.Fatalf("Unicode class digest = %s, want %s", got, unicodeMaskSHA256)
	}
	if runeClass(-1) != 0 || runeClass(utf8.MaxRune+1) != 0 {
		t.Fatal("out-of-range rune has a Unicode class")
	}
}

func TestReferenceUnicodeClasses(t *testing.T) {
	got := runReference(t, "unicode", nil)
	if len(got) != utf8.MaxRune+1 {
		t.Fatalf("reference returned %d class masks, want %d", len(got), utf8.MaxRune+1)
	}
	for i, want := range got {
		if actual := runeClass(rune(i)); actual != want {
			t.Fatalf("U+%04X class = %02x, reference = %02x", i, actual, want)
		}
	}
}

func TestSplitExamples(t *testing.T) {
	cases := []struct {
		text     string
		qwen     []string
		deepSeek []string
	}{
		{"", []string{}, []string{}},
		{"  a   b", []string{" ", " a", "  ", " b"}, []string{" ", " a", "  ", " b"}},
		{" \t\r\n  a!!\r\n\t", []string{" \t\r\n", " ", " a", "!!\r\n", "\t"},
			[]string{" \t\r\n", " ", " a", "!!\r\n", "\t"}},
		{"a123456你好あア!ABC_日本語",
			[]string{"a", "1", "2", "3", "4", "5", "6", "你好あア", "!ABC", "_日本語"},
			[]string{"a", "123", "456", "你好あア", "!ABC", "_", "日本語"}},
		{"\x00a\u0301🙂𝟝", []string{"\x00a\u0301", "🙂", "𝟝"},
			[]string{"\x00a\u0301", "🙂", "𝟝"}},
		{"I'm\u0301 don't WE'RE 'LL 'x", []string{"I", "'m", "\u0301", " don", "'t", " WE", "'RE", " '", "LL", " '", "x"},
			[]string{"I", "'m", "\u0301", " don", "'t", " WE", "'RE", " '", "LL", " '", "x"}},
	}
	for _, tc := range cases {
		for _, variant := range []struct {
			name  string
			split func(string) ([]Span, error)
			want  []string
		}{
			{"qwen", Qwen35, tc.qwen},
			{"deepseek", DeepSeek3, tc.deepSeek},
		} {
			spans, err := variant.split(tc.text)
			if err != nil {
				t.Fatalf("%s(%q): %v", variant.name, tc.text, err)
			}
			got := piecesForTest(t, tc.text, spans)
			if !reflect.DeepEqual(got, variant.want) {
				t.Errorf("%s(%q) = %q, want %q", variant.name, tc.text, got, variant.want)
			}
		}
	}
}

func TestInvalidUTF8(t *testing.T) {
	for _, text := range []string{"\xff", "x\xc3", "\xed\xa0\x80", "\xf0\x80\x80\x80"} {
		for _, split := range []func(string) ([]Span, error){Qwen35, DeepSeek3} {
			if spans, err := split(text); !errors.Is(err, ErrInvalidUTF8) || spans != nil {
				t.Fatalf("invalid input %q: got spans %v, error %v", text, spans, err)
			}
		}
	}
}

func TestReferenceSplits(t *testing.T) {
	inputs := referenceCorpus()
	data, err := json.Marshal(inputs)
	if err != nil {
		t.Fatal(err)
	}
	var expected []struct {
		Qwen     [][2]int `json:"qwen"`
		DeepSeek [][2]int `json:"deepseek"`
	}
	if err := json.Unmarshal(runReference(t, "split", data), &expected); err != nil {
		t.Fatalf("decode reference splits: %v", err)
	}
	if len(expected) != len(inputs) {
		t.Fatalf("reference returned %d inputs, want %d", len(expected), len(inputs))
	}
	for i, text := range inputs {
		for _, variant := range []struct {
			name  string
			split func(string) ([]Span, error)
			want  [][2]int
		}{
			{"qwen", Qwen35, expected[i].Qwen},
			{"deepseek", DeepSeek3, expected[i].DeepSeek},
		} {
			spans, err := variant.split(text)
			if err != nil {
				t.Fatalf("%s input %d %q: %v", variant.name, i, text, err)
			}
			piecesForTest(t, text, spans)
			if len(spans) != len(variant.want) {
				t.Fatalf("%s input %d %q: spans %v, reference %v", variant.name, i, text, spans, variant.want)
			}
			for j, span := range spans {
				if span.Start != variant.want[j][0] || span.End != variant.want[j][1] {
					t.Fatalf("%s input %d %q: spans %v, reference %v", variant.name, i, text, spans, variant.want)
				}
			}
		}
	}
	t.Logf("matched Python Qwen and DeepSeek splits on %d inputs", len(inputs))
}

func piecesForTest(t *testing.T, text string, spans []Span) []string {
	t.Helper()
	out := make([]string, 0, len(spans))
	pos := 0
	for _, span := range spans {
		if span.Start != pos || span.End <= span.Start || span.End > len(text) {
			t.Fatalf("invalid partition of %q at %+v after byte %d", text, span, pos)
		}
		piece := text[span.Start:span.End]
		if !utf8.ValidString(piece) {
			t.Fatalf("span %+v cuts a UTF-8 code point in %q", span, text)
		}
		out = append(out, piece)
		pos = span.End
	}
	if pos != len(text) {
		t.Fatalf("partition ends at byte %d, input length %d", pos, len(text))
	}
	return out
}

func referenceCorpus() []string {
	inputs := []string{
		"", "a", "\x00", "\r\n", " \t \n \vX", "\u0301", "\u0301\u0300a",
		"can't I'M WE'Re 've 'll 'd 'r 'x", " a  b   !\r\n  x",
		"123456789", "１٢𝟡㊿1234", "你好あア\u3040\u309f\u30a0\u30ff",
		"!\n_A  !ASCII-日本語𠀀", "👩‍💻️🇨🇦", "\u2028\u0085\u00a0\u200b",
	}
	// Probe both sides of each change in any reference Unicode property.
	// Surrogates have no UTF-8 encoding and are covered by the class test.
	for _, boundary := range unicodeBoundaries {
		pos := rune(boundary >> 6)
		for _, r := range []rune{pos - 1, pos} {
			if r < 0 || r > utf8.MaxRune || r >= 0xd800 && r <= 0xdfff {
				continue
			}
			inputs = append(inputs, string([]rune{'x', r, 'a', ' ', r, '\n', '1'}))
		}
	}
	// Exhaust the interactions of representative classes and ASCII pattern
	// syntax in three-character strings, including whitespace backtracking.
	alphabet3 := []rune("arElS'0 \t\r\n!\u0301\u200b中あ🙂\x00")
	for _, a := range alphabet3 {
		for _, b := range alphabet3 {
			for _, c := range alphabet3 {
				inputs = append(inputs, string([]rune{a, b, c}))
			}
		}
	}
	// Mix runs, prefixes, line endings, controls and Unicode classes in a
	// fixed-seed corpus. This deliberately uses no external data or model.
	rng := rand.New(rand.NewSource(1701))
	alphabet := []rune("aZzREm's' !_,\t\n\r\v\u0085\u00a0\u0301\u093e\u200d\u2028\u3040あア中龥𠀀🙂𝟡①٠１２\u1c89")
	for i := 0; i < 3000; i++ {
		var b strings.Builder
		for n := rng.Intn(50); n > 0; n-- {
			if rng.Intn(5) == 0 {
				r := rune(rng.Intn(utf8.MaxRune + 1))
				if r >= 0xd800 && r <= 0xdfff {
					r = '\ufffd'
				}
				b.WriteRune(r)
			} else {
				b.WriteRune(alphabet[rng.Intn(len(alphabet))])
			}
		}
		inputs = append(inputs, b.String())
	}
	return inputs
}

func runReference(t *testing.T, mode string, input []byte) []byte {
	t.Helper()
	python, err := referencePython(filepath.Join("..", ".."), os.Getenv("STRATA_REFERENCE_PYTHON"), runtime.GOOS)
	if err != nil {
		t.Fatalf("reference Python: %v", err)
	}
	if python == "" {
		t.Skip("reference Python unavailable; set STRATA_REFERENCE_PYTHON to a Python with regex 2026.9.10")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, filepath.Join("testdata", "reference.py"), mode)
	cmd.Stdin = bytes.NewReader(input)
	out, err := cmd.Output()
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			t.Fatalf("reference %s: %v\n%s", mode, err, exit.Stderr)
		}
		t.Fatal(fmt.Errorf("reference %s: %w", mode, err))
	}
	return out
}

func referencePython(root, override, goos string) (string, error) {
	if override != "" {
		return override, nil
	}
	unix := filepath.Join(root, ".venv", "bin", "python")
	windows := filepath.Join(root, ".venv", "Scripts", "python.exe")
	candidates := []string{unix, windows}
	if goos == "windows" {
		candidates[0], candidates[1] = candidates[1], candidates[0]
	}
	for _, candidate := range candidates {
		_, err := os.Stat(candidate)
		if err == nil {
			return candidate, nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
	}
	return "", nil
}

func TestReferencePythonSelection(t *testing.T) {
	cases := []struct {
		name     string
		goos     string
		files    []string
		override string
		want     string
	}{
		{"missing", "linux", nil, "", ""},
		{"unix", "linux", []string{"bin/python"}, "", "bin/python"},
		{"windows", "windows", []string{"Scripts/python.exe"}, "", "Scripts/python.exe"},
		{"unix first", "linux", []string{"bin/python", "Scripts/python.exe"}, "", "bin/python"},
		{"windows first", "windows", []string{"bin/python", "Scripts/python.exe"}, "", "Scripts/python.exe"},
		{"windows fallback", "linux", []string{"Scripts/python.exe"}, "", "Scripts/python.exe"},
		{"unix fallback", "windows", []string{"bin/python"}, "", "bin/python"},
		{"override", "windows", []string{"bin/python", "Scripts/python.exe"}, "custom-python", "custom-python"},
		{"missing override", "linux", nil, "custom-python", "custom-python"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			for _, name := range tc.files {
				path := filepath.Join(root, ".venv", filepath.FromSlash(name))
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, nil, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			want := tc.want
			if want != "" && tc.override == "" {
				want = filepath.Join(root, ".venv", filepath.FromSlash(want))
			}
			got, err := referencePython(root, tc.override, tc.goos)
			if err != nil || got != want {
				t.Fatalf("referencePython() = %q, %v; want %q, nil", got, err, want)
			}
		})
	}
}
