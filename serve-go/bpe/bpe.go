// Package bpe implements byte-level ranked BPE independently of pre-tokenization.
package bpe

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// Splitter partitions a plain text stretch into nonempty, consecutive pieces.
// It must preserve the input exactly. Callers supply the model's pre-tokenizer.
type Splitter func(string) []string

// Span protects literal starts in [Start, End), measured in UTF-8 byte offsets.
type Span struct{ Start, End int }

type pair struct{ left, right uint32 }
type literal struct {
	text    string
	id, typ int
}

// Tokenizer is immutable after construction and safe for concurrent use if its
// supplied Splitter is also safe for concurrent use.
type Tokenizer struct {
	tokens   []string
	special  []bool
	ids      map[string]uint32
	ranks    map[pair]uint32
	literals []literal
	split    Splitter
}

// VocabularySize returns the number of token IDs.
func (t *Tokenizer) VocabularySize() int { return len(t.tokens) }

// ControlLiterals returns a copy of the special-token texts.
func (t *Tokenizer) ControlLiterals() []string {
	out := make([]string, len(t.literals))
	for i, literal := range t.literals {
		out[i] = literal.text
	}
	return out
}

var byteChars [256]string
var charBytes map[rune]byte

func init() {
	charBytes = make(map[rune]byte, 256)
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = rune(256 + n)
			n++
		}
		byteChars[b] = string(r)
		charBytes[r] = byte(b)
	}
}

// New validates the vocabulary and ordered merge rules. A nil types slice means
// no special tokens. Merge rules follow the pack's "left right" format.
func New(tokens []string, merges []string, types []int, split Splitter) (*Tokenizer, error) {
	if split == nil {
		return nil, fmt.Errorf("nil splitter")
	}
	if len(types) != 0 && len(types) != len(tokens) {
		return nil, fmt.Errorf("token type count does not match vocabulary")
	}
	if uint64(len(tokens)) > uint64(^uint32(0)) || uint64(len(merges)) > uint64(^uint32(0)) {
		return nil, fmt.Errorf("vocabulary or merge count exceeds uint32")
	}
	t := &Tokenizer{tokens: append([]string(nil), tokens...), ids: make(map[string]uint32, len(tokens)), ranks: make(map[pair]uint32, len(merges)), split: split}
	if len(types) != 0 {
		t.special = make([]bool, len(types))
	}
	for i, s := range tokens {
		if _, ok := t.ids[s]; ok {
			return nil, fmt.Errorf("duplicate token %q", s)
		}
		t.ids[s] = uint32(i)
		if len(types) != 0 && (types[i] == 3 || types[i] == 4) {
			if s == "" {
				return nil, fmt.Errorf("empty special token")
			}
			t.special[i] = true
			t.literals = append(t.literals, literal{s, i, types[i]})
		}
	}
	sort.SliceStable(t.literals, func(i, j int) bool { return len(t.literals[i].text) > len(t.literals[j].text) })
	for i, rule := range merges {
		p := strings.Split(rule, " ")
		if len(p) != 2 {
			return nil, fmt.Errorf("merge %d is not a pair", i)
		}
		left, ok := t.ids[p[0]]
		if !ok {
			return nil, fmt.Errorf("merge %d has unknown left token", i)
		}
		right, ok := t.ids[p[1]]
		if !ok {
			return nil, fmt.Errorf("merge %d has unknown right token", i)
		}
		t.ranks[pair{left, right}] = uint32(i) // matches the reference's last rank for duplicate pairs
	}
	return t, nil
}

// Load reads the vocabulary, merge list and token types from a pack directory.
func Load(dir string, split Splitter) (*Tokenizer, error) {
	read := func(name string, v any) error {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		if err = json.Unmarshal(b, v); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	}
	var vocab map[string]int
	if err := read("vocab.json", &vocab); err != nil {
		return nil, err
	}
	tokens := make([]string, len(vocab))
	seen := make([]bool, len(vocab))
	for s, id := range vocab {
		if id < 0 || id >= len(tokens) || seen[id] {
			return nil, fmt.Errorf("invalid vocabulary id %d", id)
		}
		tokens[id], seen[id] = s, true
	}
	var types []int
	if err := read("token_type.json", &types); err != nil {
		return nil, err
	}
	b, err := os.ReadFile(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	merges := strings.Split(string(b), "\n")
	return New(tokens, merges, types, split)
}

type candidate struct {
	rank, pos, right    int
	leftText, rightText string
}
type candidates []candidate

func (h candidates) Len() int { return len(h) }
func (h candidates) Less(i, j int) bool {
	return h[i].rank < h[j].rank || h[i].rank == h[j].rank && h[i].pos < h[j].pos
}
func (h candidates) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candidates) Push(x any)   { *h = append(*h, x.(candidate)) }
func (h *candidates) Pop() any     { a := *h; x := a[len(a)-1]; *h = a[:len(a)-1]; return x }

func (t *Tokenizer) encodePiece(piece string) ([]int, error) {
	n := len(piece)
	if n == 0 {
		return nil, nil
	}
	parts := make([]string, n)
	next := make([]int, n)
	prev := make([]int, n)
	for i := range parts {
		parts[i] = byteChars[piece[i]]
		next[i] = i + 1
		prev[i] = i - 1
	}
	next[n-1] = -1
	h := &candidates{}
	add := func(i int) {
		if i < 0 || next[i] < 0 {
			return
		}
		j := next[i]
		left, leftOK := t.ids[parts[i]]
		right, rightOK := t.ids[parts[j]]
		if r, ok := t.ranks[pair{left, right}]; leftOK && rightOK && ok {
			heap.Push(h, candidate{int(r), i, j, parts[i], parts[j]})
		}
	}
	for i := 0; i < n-1; i++ {
		add(i)
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(candidate)
		i, j := c.pos, c.right
		if next[i] != j || parts[i] != c.leftText || parts[j] != c.rightText {
			continue
		}
		parts[i] += parts[j]
		parts[j] = ""
		next[i] = next[j]
		if next[i] >= 0 {
			prev[next[i]] = i
		}
		add(prev[i])
		add(i)
	}
	out := make([]int, 0, n)
	for i := 0; i >= 0; i = next[i] {
		id, ok := t.ids[parts[i]]
		if !ok {
			return nil, fmt.Errorf("BPE produced token outside vocabulary: %q", parts[i])
		}
		out = append(out, int(id))
	}
	return out, nil
}

// Encode matches type-4 literals always and type-3 literals when parseSpecial
// is true. Protected spans use byte offsets, unlike Python's character offsets.
func (t *Tokenizer) Encode(text string, parseSpecial bool, plain []Span) ([]int, error) {
	return t.encode(text, parseSpecial, plain)
}

// EncodeWithBudget limits the total UTF-8 input bytes before tokenization.
// The limit includes special literals, and bounds the size of each BPE piece;
// it is not a precise memory limit and does not bound splitter allocations.
// Encode retains its unlimited behavior. A negative limit is invalid.
func (t *Tokenizer) EncodeWithBudget(text string, parseSpecial bool, plain []Span, maxInputBytes int) ([]int, error) {
	if maxInputBytes < 0 {
		return nil, fmt.Errorf("negative tokenization budget")
	}
	if len(text) > maxInputBytes {
		return nil, fmt.Errorf("tokenization input is %d bytes, exceeds budget of %d", len(text), maxInputBytes)
	}
	return t.encode(text, parseSpecial, plain)
}

func (t *Tokenizer) encode(text string, parseSpecial bool, plain []Span) ([]int, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("text is not UTF-8")
	}
	spans := append([]Span(nil), plain...)
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].Start != spans[j].Start {
			return spans[i].Start < spans[j].Start
		}
		return spans[i].End < spans[j].End
	})
	spanIndex, plainEnd := 0, 0
	out := []int{}
	encodePlain := func(s string) error {
		if s == "" {
			return nil
		}
		pieces := t.split(s)
		if strings.Join(pieces, "") != s {
			return fmt.Errorf("splitter did not preserve input")
		}
		for _, p := range pieces {
			ids, err := t.encodePiece(p)
			if err != nil {
				return err
			}
			out = append(out, ids...)
		}
		return nil
	}
	pos := 0
	for i := 0; i < len(text); {
		var match *literal
		for k := range t.literals {
			l := &t.literals[k]
			if (parseSpecial || l.typ == 4) && strings.HasPrefix(text[i:], l.text) {
				match = l
				break
			}
		}
		if match == nil {
			_, size := utf8.DecodeRuneInString(text[i:])
			i += size
			continue
		}
		for spanIndex < len(spans) && spans[spanIndex].Start <= i {
			if spans[spanIndex].End > plainEnd {
				plainEnd = spans[spanIndex].End
			}
			spanIndex++
		}
		protected := i < plainEnd
		end := i + len(match.text)
		if !protected {
			if err := encodePlain(text[pos:i]); err != nil {
				return nil, err
			}
			out = append(out, match.id)
			pos = end
		}
		i = end // reference regex advances even for a protected match
	}
	if err := encodePlain(text[pos:]); err != nil {
		return nil, err
	}
	return out, nil
}

// TokenBytes returns raw token bytes, including literal UTF-8 for special tokens.
func (t *Tokenizer) TokenBytes(id int) ([]byte, error) {
	if id < 0 || id >= len(t.tokens) {
		return nil, fmt.Errorf("token id %d out of range", id)
	}
	s := t.tokens[id]
	if len(t.special) > 0 && t.special[id] {
		return []byte(s), nil
	}
	b := make([]byte, 0, len(s))
	for _, r := range s {
		v, ok := charBytes[r]
		if !ok {
			return nil, fmt.Errorf("token %d outside byte alphabet", id)
		}
		b = append(b, v)
	}
	return b, nil
}

// Decode joins token bytes and replaces malformed UTF-8 as Python does.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b []byte
	for _, id := range ids {
		part, err := t.TokenBytes(id)
		if err != nil {
			return "", err
		}
		b = append(b, part...)
	}
	var out strings.Builder
	for i := 0; i < len(b); {
		_, size := utf8.DecodeRune(b[i:])
		if size != 1 || b[i] < utf8.RuneSelf {
			out.Write(b[i : i+size])
			i += size
			continue
		}
		// Consume the valid prefix of an incomplete sequence as one error.
		// Invalid leading bytes and disallowed second bytes stand alone.
		lead := b[i]
		width := 0
		switch {
		case lead >= 0xc2 && lead <= 0xdf:
			width = 2
		case lead >= 0xe0 && lead <= 0xef:
			width = 3
		case lead >= 0xf0 && lead <= 0xf4:
			width = 4
		}
		n := 1
		for n < width && i+n < len(b) {
			c := b[i+n]
			if c < 0x80 || c > 0xbf || n == 1 &&
				(lead == 0xe0 && c < 0xa0 || lead == 0xed && c > 0x9f ||
					lead == 0xf0 && c < 0x90 || lead == 0xf4 && c > 0x8f) {
				break
			}
			n++
		}
		out.WriteRune(utf8.RuneError)
		i += n
	}
	return out.String(), nil
}
