package bpe

import (
	"encoding/json"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func identity(s string) []string { return []string{s} }

func synthetic(t *testing.T) *Tokenizer {
	t.Helper()
	tokens := make([]string, 256)
	for i := range tokens {
		tokens[i] = byteChars[i]
	}
	tokens = append(tokens, "ab", "bc", "abc", "<x>", "<xy>", "｜x｜")
	types := make([]int, len(tokens))
	types[259] = 3
	types[260] = 4
	types[261] = 4
	tok, err := New(tokens, []string{"b c", "a b", "a bc"}, types, identity)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestRankSpecialSpansAndBytes(t *testing.T) {
	tok := synthetic(t)
	cases := []struct {
		text    string
		special bool
		spans   []Span
		want    []int
	}{
		{"abc", false, nil, []int{258}},
		{"<x><xy>｜x｜", false, nil, []int{60, 120, 62, 260, 261}},
		{"<x><xy>｜x｜", true, nil, []int{259, 260, 261}},
		{"é<x><xy>", true, []Span{{2, 5}}, []int{195, 169, 60, 120, 62, 260}},
	}
	for _, c := range cases {
		got, err := tok.Encode(c.text, c.special, c.spans)
		if err != nil || !reflect.DeepEqual(got, c.want) {
			t.Fatalf("%q: %v, %v; want %v", c.text, got, err, c.want)
		}
		decoded, err := tok.Decode(got)
		if err != nil || decoded != c.text {
			t.Fatalf("round trip: %q %v", decoded, err)
		}
	}
	all := make([]int, 256)
	for i := range all {
		all[i] = i
		b, err := tok.TokenBytes(i)
		if err != nil || len(b) != 1 || b[0] != byte(i) {
			t.Fatalf("byte %d: %v %v", i, b, err)
		}
	}
}

func TestErrorsAndConcurrency(t *testing.T) {
	tok := synthetic(t)
	if _, err := New([]string{"a", "a"}, nil, nil, identity); err == nil {
		t.Fatal("duplicate accepted")
	}
	if _, err := New([]string{"a"}, []string{"a b"}, nil, identity); err == nil {
		t.Fatal("unknown merge accepted")
	}
	if _, err := tok.TokenBytes(-1); err == nil {
		t.Fatal("invalid id accepted")
	}
	bad, _ := New([]string{"a"}, nil, nil, func(string) []string { return []string{"b"} })
	if _, err := bad.Encode("a", false, nil); err == nil {
		t.Fatal("bad splitter accepted")
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if _, err := tok.Encode("abc<x>é", true, nil); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}

func TestDecodeMalformedUTF8(t *testing.T) {
	tok := synthetic(t)
	cases := []struct {
		name string
		ids  []int
		want string
	}{
		{"consecutive invalid bytes", []int{0xff, 0xfe, 0x80}, "\ufffd\ufffd\ufffd"},
		{"truncated two-byte", []int{0xc2}, "\ufffd"},
		{"truncated three-byte", []int{0xe2, 0x82}, "\ufffd"},
		{"truncated four-byte", []int{0xf0, 0x9f, 0x92}, "\ufffd"},
		{"interrupted prefix", []int{0xe2, 0x82, 'A', 0xff}, "\ufffdA\ufffd"},
		{"invalid second bytes", []int{0xe0, 0x80, 0xed, 0xa0, 0xf0, 0x80, 0xf4, 0x90}, "\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd"},
		{"valid across tokens", []int{0xf0, 0x9f, 0x98, 0x80}, "😀"},
		{"literal replacement rune", []int{0xef, 0xbf, 0xbd, 0xff}, "\ufffd\ufffd"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Decode(tc.ids)
			if err != nil || got != tc.want {
				t.Fatalf("Decode(%x) = %q, %v; want %q", tc.ids, got, err, tc.want)
			}
		})
	}
}

func TestProtectedSpansDifferential(t *testing.T) {
	tokens := make([]string, 256)
	for i := range tokens {
		tokens[i] = byteChars[i]
	}
	tokens = append(tokens, "aba", "ba", "<x>", "<xy>")
	types := make([]int, len(tokens))
	types[256], types[257], types[258], types[259] = 4, 4, 3, 4
	tok, err := New(tokens, nil, types, identity)
	if err != nil {
		t.Fatal(err)
	}
	// Independent straightforward reference: find the next leftmost literal,
	// choose the longest there, then check every span at its start.
	reference := func(text string, special bool, spans []Span) []int {
		out := []int{}
		pos, scan := 0, 0
		for scan < len(text) {
			start, id, length := len(text), -1, 0
			for i := 256; i < len(tokens); i++ {
				if types[i] == 3 && !special {
					continue
				}
				at := strings.Index(text[scan:], tokens[i])
				if at >= 0 {
					at += scan
					if at < start || at == start && len(tokens[i]) > length {
						start, id, length = at, i, len(tokens[i])
					}
				}
			}
			if id < 0 {
				break
			}
			protected := false
			for _, span := range spans {
				protected = protected || span.Start <= start && start < span.End
			}
			if !protected {
				for _, b := range []byte(text[pos:start]) {
					out = append(out, int(b))
				}
				out = append(out, id)
				pos = start + length
			}
			scan = start + length
		}
		for _, b := range []byte(text[pos:]) {
			out = append(out, int(b))
		}
		return out
	}
	rng := rand.New(rand.NewSource(537))
	for _, text := range []string{"ababa<x><xy>aba", "éaba<x>ba<xy>"} {
		for n := 0; n < 200; n++ {
			spans := make([]Span, rng.Intn(8))
			for i := range spans {
				spans[i] = Span{rng.Intn(len(text)+5) - 2, rng.Intn(len(text)+5) - 2}
			}
			original := append([]Span(nil), spans...)
			for _, special := range []bool{false, true} {
				got, err := tok.Encode(text, special, spans)
				want := reference(text, special, spans)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("%q special=%v spans=%v: got %v, %v; want %v", text, special, spans, got, err, want)
				}
			}
			if !reflect.DeepEqual(spans, original) && len(spans) != 0 {
				t.Fatal("input spans mutated")
			}
		}
	}
}

func TestEncodeBudget(t *testing.T) {
	tok := synthetic(t)
	for _, tc := range []struct {
		text  string
		limit int
		ok    bool
	}{
		{"", 0, true}, {"abc", 3, true}, {"abc", 2, false},
		{"é", 1, false}, {"é", 2, true}, {"<x>", 2, false},
		{"", -1, false},
	} {
		got, err := tok.EncodeWithBudget(tc.text, true, nil, tc.limit)
		if (err == nil) != tc.ok {
			t.Fatalf("%q limit %d: %v", tc.text, tc.limit, err)
		}
		if tc.ok {
			want, err := tok.Encode(tc.text, true, nil)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("%q: got %v, want %v (%v)", tc.text, got, want, err)
			}
		}
	}
	called := false
	bounded, err := New([]string{"a"}, nil, nil, func(s string) []string {
		called = true
		return []string{s}
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bounded.EncodeWithBudget("aa", false, nil, 1); err == nil || called {
		t.Fatal("over-budget input reached splitter")
	}
}

// This uses only a small synthetic vocabulary and the checkout's Python reference.
func TestSyntheticPythonDifferential(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	python := filepath.Join(root, ".venv", "bin", "python")
	if _, err := os.Stat(python); os.IsNotExist(err) {
		t.Skip("checkout .venv/bin/python is absent")
	} else if err != nil {
		t.Fatal(err)
	}
	type sample struct {
		Text    string
		Special bool
		Plain   []Span
		Merges  []string
		Bytes   []int
	}
	type result struct {
		IDs     []int
		Decoded string
	}
	tokens := make([]string, 256)
	for i := range tokens {
		tokens[i] = byteChars[i]
	}
	tokens = append(tokens, "ab", "bc", "abc", "aba", "ba", "<x>")
	types := make([]int, len(tokens))
	types[259], types[260], types[261] = 4, 4, 3
	rng := rand.New(rand.NewSource(310537))
	parts := []string{"a", "b", "c", "abc", "aba", "ba", "<x>", " "}
	baseMerges := []string{"a b", "b c", "a bc"}
	samples := make([]sample, 120)
	for i := range samples {
		s := &samples[i]
		var text strings.Builder
		for j := 0; j < 2+rng.Intn(12); j++ {
			text.WriteString(parts[rng.Intn(len(parts))])
		}
		s.Text, s.Special = text.String(), rng.Intn(2) == 0
		// Deliberately unsorted, overlapping spans, plus random boundaries.
		s.Plain = []Span{{3, 8}, {0, 5}}
		for j := 0; j < 3; j++ {
			s.Plain = append(s.Plain, Span{rng.Intn(len(s.Text) + 1), rng.Intn(len(s.Text) + 1)})
		}
		s.Merges = append([]string(nil), baseMerges...)
		rng.Shuffle(len(s.Merges), func(i, j int) { s.Merges[i], s.Merges[j] = s.Merges[j], s.Merges[i] })
		s.Bytes = make([]int, 1+rng.Intn(24))
		for j := range s.Bytes {
			s.Bytes[j] = rng.Intn(256)
		}
	}
	input, err := json.Marshal(struct {
		Tokens  []string
		Types   []int
		Samples []sample
	}{tokens, types, samples})
	if err != nil {
		t.Fatal(err)
	}
	script := `import json, sys
sys.path.insert(0, sys.argv[1] + "/tools")
import strata_tokenizer as st
data = json.load(sys.stdin)
out = []
for s in data["Samples"]:
    tok = st.Tokenizer(data["Tokens"], s["Merges"], data["Types"])
    tok._split = lambda text: [text]
    spans = [(p["Start"], p["End"]) for p in s["Plain"]]
    out.append({"IDs": tok.encode(s["Text"], s["Special"], spans),
                "Decoded": tok.decode(s["Bytes"])})
json.dump(out, sys.stdout)
`
	cmd := exec.Command(python, "-c", script, root)
	cmd.Stdin = strings.NewReader(string(input))
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python reference: %v: %s", err, output)
	}
	var expected []result
	if err := json.Unmarshal(output, &expected); err != nil {
		t.Fatal(err)
	}
	if len(expected) != len(samples) {
		t.Fatalf("reference returned %d results for %d samples", len(expected), len(samples))
	}
	for i, s := range samples {
		tok, err := New(tokens, s.Merges, types, identity)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tok.Encode(s.Text, s.Special, s.Plain)
		if err != nil || !reflect.DeepEqual(ids, expected[i].IDs) {
			t.Fatalf("sample %d encoding: %v, %v; want %v; input %+v", i, ids, err, expected[i].IDs, s)
		}
		decoded, err := tok.Decode(s.Bytes)
		if err != nil || decoded != expected[i].Decoded {
			t.Fatalf("sample %d decoding %x: %q, %v; want %q", i, s.Bytes, decoded, err, expected[i].Decoded)
		}
	}
}

// STRATA_BPE_PACK and STRATA_BPE_PYTHON opt into a differential run against
// tools/strata_tokenizer.py. The injected identity splitter isolates BPE.
func TestReferencePack(t *testing.T) {
	dir, python := os.Getenv("STRATA_BPE_PACK"), os.Getenv("STRATA_BPE_PYTHON")
	if dir == "" || python == "" {
		t.Skip("set STRATA_BPE_PACK and STRATA_BPE_PYTHON to run reference comparison")
	}
	tok, err := Load(dir, identity)
	if err != nil {
		t.Fatal(err)
	}
	texts := []string{"abc abc", "Hello, café 你好 😀\n", strings.Repeat("abc", 120), "<think>x</think><|im_end|>", "é<|im_end|>x", "\x00\t\r\n"}
	type sample struct {
		Text    string `json:"text"`
		Special bool   `json:"special"`
		Plain   []Span `json:"plain"`
	}
	var samples []sample
	for _, s := range texts {
		for _, special := range []bool{false, true} {
			samples = append(samples, sample{s, special, nil})
		}
	}
	samples = append(samples, sample{"é<|im_end|><think>", true, []Span{{2, 12}}})
	input, _ := json.Marshal(samples)
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	script := `import json,sys,pathlib
sys.path.insert(0,sys.argv[2]+"/tools")
import strata_tokenizer as st
p=pathlib.Path(sys.argv[1])
v=json.loads((p/"vocab.json").read_text())
tokens=[None]*len(v)
for s,i in v.items(): tokens[i]=s
tok=st.Tokenizer(tokens,(p/"merges.txt").read_text().split("\n"),json.loads((p/"token_type.json").read_text()))
tok._split=lambda s:[s]
out=[]
for x in json.load(sys.stdin):
 s=x["text"]
 spans=[(len(s.encode()[:p["Start"]].decode()),len(s.encode()[:p["End"]].decode())) for p in (x["plain"] or [])]
 out.append(tok.encode(s,x["special"],spans))
json.dump(out,sys.stdout)`
	cmd := exec.Command(python, "-c", script, dir, root)
	cmd.Stdin = strings.NewReader(string(input))
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("reference: %v: %s", err, output)
	}
	var want [][]int
	if err := json.Unmarshal(output, &want); err != nil {
		t.Fatal(err)
	}
	for i, s := range samples {
		got, err := tok.Encode(s.Text, s.Special, s.Plain)
		if err != nil || !reflect.DeepEqual(got, want[i]) {
			t.Fatalf("sample %d: %v %v != %v", i, got, err, want[i])
		}
	}
}
