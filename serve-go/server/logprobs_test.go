package server

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestLogprobCollectorSpell(t *testing.T) {
	tok := sharedTokenizer(t)
	c := newLogprobCollector(tok, map[int]bool{257: true}, nil)
	for _, id := range []int{'x', 0xe2, 0x82, 0xac, 257, 'a', 'b'} {
		if err := c.Add(logprobInput{ID: id, Logprob: -1}); err != nil {
			t.Fatal(err)
		}
	}
	got := c.Spell("€")
	if len(got) != 3 || got[0].Token != "" || got[1].Token != "" || got[2].Token != "€" {
		t.Fatalf("split character: %#v", got)
	}
	if !reflect.DeepEqual(got[0].Bytes, []int{0xe2}) {
		t.Fatalf("raw bytes: %v", got[0].Bytes)
	}
	got = c.Spell("a")
	if len(got) != 1 || got[0].Token != "a" {
		t.Fatalf("pending match: %#v", got)
	}
	got = c.Spell("not present")
	if len(got) != 1 || got[0].Token != "b" {
		t.Fatalf("fallback: %#v", got)
	}
}

func TestLogprobCollectorBoundaries(t *testing.T) {
	tok := sharedTokenizer(t)
	c := newLogprobCollector(tok, nil, nil)
	for _, id := range []int{259, 'a', 'b'} {
		if err := c.Add(logprobInput{ID: id}); err != nil {
			t.Fatal(err)
		}
	}
	got := c.Spell("a")
	if len(got) != 1 || got[0].Token != "a" {
		t.Fatalf("must skip match within a token: %#v", got)
	}
	if got = c.Spell(""); len(got) != 1 || got[0].Token != "b" {
		t.Fatalf("empty text fallback: %#v", got)
	}
}

func TestLogprobCollectorAllowed(t *testing.T) {
	tok := sharedTokenizer(t)
	c := newLogprobCollector(tok, nil, []int{'a', 'b', 0xe2})
	if err := c.Add(logprobInput{ID: 'a', Logprob: -2, Alternatives: []logprobAlternative{
		{ID: 'b', Logprob: -1}, {ID: 0xe2, Logprob: -1}, {ID: 'c', Logprob: -0.1},
	}}); err != nil {
		t.Fatal(err)
	}
	got := c.Spell("a")[0]
	if len(got.TopLogprobs) != 3 || got.TopLogprobs[0].Token != "b" || got.TopLogprobs[1].Token != "�" || got.TopLogprobs[2].Token != "a" {
		t.Fatalf("filtered stable ordering: %#v", got.TopLogprobs)
	}
	sum := 0.0
	for _, alt := range got.TopLogprobs {
		sum += math.Exp(alt.Logprob)
	}
	if math.Abs(sum-1) > 1e-12 || got.Logprob != got.TopLogprobs[2].Logprob {
		t.Fatalf("normalization: %#v, sum %g", got, sum)
	}
	b, err := json.Marshal(got)
	if err != nil || !json.Valid(b) {
		t.Fatalf("JSON: %s %v", b, err)
	}
	var obj map[string]any
	if err := json.Unmarshal(b, &obj); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(obj["bytes"], []any{float64('a')}) {
		t.Fatalf("bytes must be a numeric array: %s", b)
	}
}

func TestLogprobCollectorNoSurvivorAndOwn(t *testing.T) {
	tok := sharedTokenizer(t)
	c := newLogprobCollector(tok, nil, []int{'z'})
	if err := c.Add(logprobInput{ID: 'a', Logprob: -0.3, Alternatives: []logprobAlternative{{ID: 'b', Logprob: -1}}}); err != nil {
		t.Fatal(err)
	}
	got := c.Spell("a")[0]
	if got.Logprob != -0.3 || len(got.TopLogprobs) != 0 {
		t.Fatalf("missing survivors: %#v", got)
	}
	if err := c.Add(logprobInput{ID: 'z', Logprob: -0.3}); err != nil {
		t.Fatal(err)
	}
	got = c.Spell("z")[0]
	if got.Logprob != 0 || len(got.TopLogprobs) != 1 || got.TopLogprobs[0].Logprob != 0 {
		t.Fatalf("chosen only: %#v", got)
	}
}
