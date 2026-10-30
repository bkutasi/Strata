package streamtext

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestBoundaries(t *testing.T) {
	cases := []struct {
		input     string
		stops     []string
		want, hit string
	}{
		{"alpha END beta", []string{"END"}, "alpha ", "END"},
		{"a 🌊世界!", []string{"世界"}, "a 🌊", "世界"},
		{"a </x", []string{"</s>"}, "a </x", ""},
		{"xabc!", []string{"abc", "ab"}, "x", "ab"},
		{"\xe1\x80X\xf0\x9f", nil, "\ufffdX", ""},
		{"ab", []string{"abc"}, "ab", ""},
	}
	for _, tc := range cases {
		// All partitions, including a split at every byte.
		for mask := 0; mask < 1<<(len(tc.input)-1); mask++ {
			s := NewStopMatcher(tc.stops)
			var d Decoder
			var out strings.Builder
			start := 0
			for i := 1; i <= len(tc.input); i++ {
				if i == len(tc.input) || mask&(1<<(i-1)) != 0 {
					if s.Hit() == "" {
						out.WriteString(s.Push(d.Push([]byte(tc.input[start:i]))))
					}
					start = i
				}
			}
			out.WriteString(s.Finish())
			if out.String() != tc.want || s.Hit() != tc.hit {
				// A short stop may match in an earlier chunk than a longer
				// stop at the same position; both are Python behavior.
				if tc.input == "xabc!" && out.String() == "x" && s.Hit() == "abc" {
					continue
				}
				t.Fatalf("%q partition %x: got %q hit %q", tc.input, mask, out.String(), s.Hit())
			}
			if s.Finish() != "" || s.Push("later") != "" {
				t.Fatal("stream emitted after finish")
			}
		}
	}
}

func TestHoldingAndTie(t *testing.T) {
	s := NewStopMatcher([]string{"</s>", ""})
	for _, step := range []struct{ in, out string }{
		{"a <", "a "}, {"b", "<b"}, {"c </", "c "}, {"s> d", ""}, {"more", ""},
	} {
		if got := s.Push(step.in); got != step.out {
			t.Fatalf("push %q = %q, want %q", step.in, got, step.out)
		}
	}
	if s.Hit() != "</s>" || s.Finish() != "" {
		t.Fatal("match was not terminal")
	}
	s = NewStopMatcher([]string{"abc", "ab"})
	if s.Push("zabc") != "z" || s.Hit() != "abc" {
		t.Fatal("ties must follow stop order")
	}
}

func TestFinishAndParserBoundary(t *testing.T) {
	s := NewStopMatcher([]string{"abc"})
	var d Decoder
	if got := s.Push(d.Push([]byte{'a', 'b', 0xe2, 0x82})); got != "" || !d.Pending() {
		t.Fatalf("initial push = %q pending = %v", got, d.Pending())
	}
	if got := s.Finish(); got != "ab" || s.Hit() != "" {
		t.Fatalf("finish = %q, hit = %q", got, s.Hit())
	}
	for _, kind := range []string{"tool", "reasoning"} {
		t.Run(kind, func(t *testing.T) {
			s := NewStopMatcher([]string{"abc"})
			if got := s.Push("before ab"); got != "before " {
				t.Fatalf("push = %q", got)
			}
			if got := s.Boundary(); got != "ab" {
				t.Fatalf("boundary = %q", got)
			}
			if got := s.Push("c after"); got != "c after" || s.Hit() != "" {
				t.Fatalf("after %s: %q hit %q", kind, got, s.Hit())
			}
		})
	}
	s = NewStopMatcher([]string{"€!"})
	d = Decoder{}
	s.Push(d.Push([]byte{0xe2, 0x82}))
	if s.Boundary() != "" || !d.Pending() {
		t.Fatal("boundary changed pending decoder bytes")
	}
	if got := s.Push(d.Push([]byte{0xac, '!'})); got != "" || s.Hit() != "€!" {
		t.Fatalf("decoder did not survive boundary: %q hit %q", got, s.Hit())
	}
	if s.Boundary() != "" {
		t.Fatal("boundary emitted after hit")
	}
}

func TestParserEventRouting(t *testing.T) {
	// Simulated parser events: stop text in non-content must pass untouched,
	// and held content must be flushed before the intervening event.
	for _, kind := range []string{"reasoning", "tool"} {
		t.Run(kind, func(t *testing.T) {
			var d Decoder
			m := NewStopMatcher([]string{"STOP"})
			type event struct{ kind, text string }
			input := []event{{"content", "before ST"}, {kind, "STOP"}, {"content", "OP after "}, {"content", "STOP hidden"}}
			var output []event
			for _, ev := range input {
				text := d.Push([]byte(ev.text)) // representative parser emits this event
				if ev.kind != "content" {
					if held := m.Boundary(); held != "" {
						output = append(output, event{"content", held})
					}
					output = append(output, event{ev.kind, text})
				} else if safe := m.Push(text); safe != "" {
					output = append(output, event{"content", safe})
				}
			}
			want := []event{{"content", "before "}, {"content", "ST"}, {kind, "STOP"}, {"content", "OP after "}}
			if len(output) != len(want) {
				t.Fatalf("events = %#v", output)
			}
			for i := range want {
				if output[i] != want[i] {
					t.Fatalf("events = %#v", output)
				}
			}
			if m.Hit() != "STOP" {
				t.Fatalf("hit = %q", m.Hit())
			}
		})
	}
}

type diffCase struct {
	Stops      []string `json:"stops"`
	Chunks     [][]byte `json:"chunks"`
	Boundaries []bool   `json:"boundaries"`
}

type diffResult struct {
	Out     []string `json:"out"`
	Pending []bool   `json:"pending"`
	Hit     string   `json:"hit"`
	Finish  string   `json:"finish"`
}

// Differential scope: decoder output routed entirely as content, with synthetic
// non-content boundaries; this does not exercise Python's OutputParser.
func TestPythonDifferential(t *testing.T) {
	python := "../../.venv/bin/python"
	if _, err := os.Stat(python); os.IsNotExist(err) {
		t.Skip("repository .venv unavailable")
	} else if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	cases := []diffCase{
		{Stops: []string{"\ufffd", "x"}, Chunks: [][]byte{{0xe1, 0x80}, {'x'}}},
		{Stops: []string{"🌊", "abc"}, Chunks: [][]byte{[]byte("x🌊abc")}},
		{Stops: []string{"abc", "\ufffd"}, Chunks: [][]byte{[]byte("ab"), {0xe2, 0x82}}},
		{Stops: []string{"abc"}, Chunks: [][]byte{[]byte("ab"), []byte("c")}, Boundaries: []bool{true, false}},
	}
	for i := 0; i < 1500; i++ {
		c := diffCase{Stops: []string{"ab", "🌊", "\ufffdx", "x\ufffd", "abc"}}
		for j := 0; j < 1+rng.Intn(8); j++ {
			chunk := make([]byte, rng.Intn(7))
			for k := range chunk {
				chunk[k] = byte(rng.Intn(256))
			}
			c.Chunks = append(c.Chunks, chunk)
			c.Boundaries = append(c.Boundaries, rng.Intn(4) == 0)
		}
		cases = append(cases, c)
	}
	data, err := json.Marshal(cases)
	if err != nil {
		t.Fatal(err)
	}
	const script = `
import base64, json, sys
sys.path.insert(0, "../..")
from serve.server import Detokenizer, StopMatcher
class Tok:
    def token_bytes(self, t): return t
results = []
for case in json.load(sys.stdin):
    d, m = Detokenizer(Tok()), StopMatcher(case["stops"])
    out, pending = [], []
    for i, chunk in enumerate(case["chunks"]):
        if m.hit is None:
            text = m.push(d.push(base64.b64decode(chunk or "")))
            if case["boundaries"] and case["boundaries"][i] and m.hit is None:
                text += m.flush()  # cut() before a non-content parser event
            out.append(text)
        else:
            out.append("")
        pending.append(d.pending())
    tail = m.flush() if m.hit is None else ""  # server finish: no decoder finalization
    results.append(dict(out=out, pending=pending, hit=m.hit or "", finish=tail))
json.dump(results, sys.stdout, ensure_ascii=True)
`
	cmd := exec.Command(python, "-c", script)
	cmd.Stdin = bytes.NewReader(data)
	output, err := cmd.Output()
	if err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			t.Fatalf("Python: %v\n%s", err, e.Stderr)
		}
		t.Fatal(err)
	}
	var expected []diffResult
	if err := json.Unmarshal(output, &expected); err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		s := NewStopMatcher(c.Stops)
		var d Decoder
		got := diffResult{Out: []string{}, Pending: []bool{}}
		for j, chunk := range c.Chunks {
			text := ""
			if s.Hit() == "" {
				text = s.Push(d.Push(chunk))
			}
			if len(c.Boundaries) > 0 && c.Boundaries[j] {
				text += s.Boundary()
			}
			got.Out = append(got.Out, text)
			got.Pending = append(got.Pending, d.Pending())
		}
		got.Finish = s.Finish()
		got.Hit = s.Hit()
		a, _ := json.Marshal(got)
		b, _ := json.Marshal(expected[i])
		if !bytes.Equal(a, b) {
			t.Fatalf("case %d %+v: got %s want %s", i, c, a, b)
		}
	}
}
