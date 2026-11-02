// Package bpe implements byte-level ranked BPE independently of pre-tokenization.
package bpe

import (
	"bufio"
	"container/heap"
	"encoding/json"
	"fmt"
	"hash/maphash"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Splitter partitions a plain text stretch into nonempty, consecutive pieces.
// It must preserve the input exactly. Callers supply the model's pre-tokenizer.
type Splitter func(string) []string

// Span protects literal starts in [Start, End), measured in UTF-8 byte offsets.
type Span struct{ Start, End int }

type rankEntry struct{ right, rank uint32 }
type mergeEntry struct{ left, right, rank uint32 }
type literal struct {
	text    string
	id, typ int
}

// Tokenizer is immutable after construction and safe for concurrent use if its
// supplied Splitter is also safe for concurrent use.
type Tokenizer struct {
	tokens       string
	shortOffsets []byte
	offsets      []uint32
	wideOffsets  []int
	special      []byte // one bit per special-token ID
	ids          []byte // 24-bit ID+1 slots; zero is empty
	wideIDs      []uint32
	seed         maphash.Seed
	rankOffsets  []uint32
	shortRanks   []byte
	ranks        []rankEntry
	compactRanks []byte
	literals     []literal
	split        Splitter
}

// VocabularySize returns the number of token IDs.
func (t *Tokenizer) VocabularySize() int {
	if t.wideOffsets != nil {
		return len(t.wideOffsets) - 1
	}
	if t.shortOffsets != nil {
		return len(t.shortOffsets)/3 - 1
	}
	return len(t.offsets) - 1
}

func offset24(data []byte, i int) uint32 {
	p := data[i*3 : i*3+3]
	return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16
}

func setOffset24(data []byte, i int, value uint32) {
	p := data[i*3 : i*3+3]
	p[0], p[1], p[2] = byte(value), byte(value>>8), byte(value>>16)
}

func (t *Tokenizer) token(id int) string {
	if t.wideOffsets != nil {
		return t.tokens[t.wideOffsets[id]:t.wideOffsets[id+1]]
	}
	if t.shortOffsets != nil {
		return t.tokens[offset24(t.shortOffsets, id):offset24(t.shortOffsets, id+1)]
	}
	return t.tokens[t.offsets[id]:t.offsets[id+1]]
}

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

// indexCapacity uses a power of two and keeps at least half the slots empty.
func indexCapacity(n int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if n < 0 || uint64(n) > uint64(^uint32(0)) {
		return 0, fmt.Errorf("vocabulary count exceeds uint32")
	}
	capacity := 1
	for capacity/2 < n {
		if capacity > maxInt/2 || capacity*2 > maxInt/4 {
			return 0, fmt.Errorf("vocabulary index size overflows int")
		}
		capacity *= 2
	}
	return capacity, nil
}

func (t *Tokenizer) indexEntry(slot int) uint32 {
	if t.wideIDs != nil {
		return t.wideIDs[slot]
	}
	i := 3 * slot
	return uint32(t.ids[i]) | uint32(t.ids[i+1])<<8 | uint32(t.ids[i+2])<<16
}

func (t *Tokenizer) setIndexEntry(slot int, value uint32) {
	if t.wideIDs != nil {
		t.wideIDs[slot] = value
		return
	}
	i := 3 * slot
	t.ids[i], t.ids[i+1], t.ids[i+2] = byte(value), byte(value>>8), byte(value>>16)
}

// findID returns the matching ID or the empty slot where it can be inserted.
// Slots store ID+1 so that even an empty token at ID zero is representable.
func (t *Tokenizer) findID(s string) (uint32, int, bool) {
	capacity := len(t.ids) / 3
	if t.wideIDs != nil {
		capacity = len(t.wideIDs)
	}
	mask := capacity - 1
	pos := int(maphash.String(t.seed, s) & uint64(mask))
	for {
		entry := t.indexEntry(pos)
		if entry == 0 {
			return 0, pos, false
		}
		id := entry - 1
		if t.token(int(id)) == s {
			return id, pos, true
		}
		pos = (pos + 1) & mask
	}
}

func (t *Tokenizer) tokenID(s string) (uint32, bool) {
	id, _, ok := t.findID(s)
	return id, ok
}

func (t *Tokenizer) rankAt(i int) rankEntry {
	if t.ranks != nil {
		return t.ranks[i]
	}
	p := t.compactRanks[i*6 : i*6+6]
	return rankEntry{
		right: uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16,
		rank:  uint32(p[3]) | uint32(p[4])<<8 | uint32(p[5])<<16,
	}
}

func (t *Tokenizer) mergeRank(left, right uint32) (uint32, bool) {
	var start, end int
	if t.shortRanks != nil {
		start, end = int(offset24(t.shortRanks, int(left))), int(offset24(t.shortRanks, int(left)+1))
	} else {
		start, end = int(t.rankOffsets[left]), int(t.rankOffsets[int(left)+1])
	}
	i := sort.Search(end-start, func(i int) bool { return t.rankAt(start+i).right >= right })
	if i == end-start {
		return 0, false
	}
	entry := t.rankAt(start + i)
	if entry.right != right {
		return 0, false
	}
	return entry.rank, true
}

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
	capacity, err := indexCapacity(len(tokens))
	if err != nil {
		return nil, err
	}
	maxInt := int(^uint(0) >> 1)
	if len(tokens) >= maxInt/(strconv.IntSize/8) {
		return nil, fmt.Errorf("vocabulary offsets size overflows int")
	}
	if len(merges) > maxInt/12 {
		return nil, fmt.Errorf("merge entries size overflows int")
	}
	total := 0
	for _, s := range tokens {
		if len(s) > maxInt-total {
			return nil, fmt.Errorf("vocabulary text size overflows int")
		}
		total += len(s)
	}
	var packed strings.Builder
	packed.Grow(total)
	var shortOffsets []byte
	var offsets []uint32
	var wideOffsets []int
	if total < 1<<24 {
		shortOffsets = make([]byte, (len(tokens)+1)*3)
	} else if uint64(total) <= uint64(^uint32(0)) {
		offsets = make([]uint32, len(tokens)+1)
	} else {
		wideOffsets = make([]int, len(tokens)+1)
	}
	for i, s := range tokens {
		packed.WriteString(s)
		if wideOffsets != nil {
			wideOffsets[i+1] = packed.Len()
		} else if shortOffsets != nil {
			setOffset24(shortOffsets, i+1, uint32(packed.Len()))
		} else {
			offsets[i+1] = uint32(packed.Len())
		}
	}
	t := &Tokenizer{tokens: packed.String(), shortOffsets: shortOffsets, offsets: offsets, wideOffsets: wideOffsets, seed: maphash.MakeSeed(), split: split}
	if len(tokens) <= 1<<24-1 {
		t.ids = make([]byte, capacity*3)
	} else {
		t.wideIDs = make([]uint32, capacity)
	}
	if len(types) != 0 {
		t.special = make([]byte, (len(types)+7)/8)
	}
	for i, s := range tokens {
		_, slot, ok := t.findID(s)
		if ok {
			return nil, fmt.Errorf("duplicate token %q", s)
		}
		t.setIndexEntry(slot, uint32(i)+1)
		if len(types) != 0 && (types[i] == 3 || types[i] == 4) {
			if s == "" {
				return nil, fmt.Errorf("empty special token")
			}
			t.special[i/8] |= 1 << (i % 8)
			t.literals = append(t.literals, literal{t.token(i), i, types[i]})
		}
	}
	sort.SliceStable(t.literals, func(i, j int) bool { return len(t.literals[i].text) > len(t.literals[j].text) })
	entries := make([]mergeEntry, len(merges))
	for i, rule := range merges {
		p := strings.Split(rule, " ")
		if len(p) != 2 {
			return nil, fmt.Errorf("merge %d is not a pair", i)
		}
		left, ok := t.tokenID(p[0])
		if !ok {
			return nil, fmt.Errorf("merge %d has unknown left token", i)
		}
		right, ok := t.tokenID(p[1])
		if !ok {
			return nil, fmt.Errorf("merge %d has unknown right token", i)
		}
		entries[i] = mergeEntry{left, right, uint32(i)}
	}
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.left != b.left {
			return a.left < b.left
		}
		if a.right != b.right {
			return a.right < b.right
		}
		return a.rank > b.rank
	})
	unique := 0
	for i, entry := range entries {
		if i == 0 || entry.left != entries[i-1].left || entry.right != entries[i-1].right {
			unique++
		}
	}
	if unique < 1<<24 {
		t.shortRanks = make([]byte, (len(tokens)+1)*3)
	} else {
		t.rankOffsets = make([]uint32, len(tokens)+1)
	}
	if len(tokens) <= 1<<24 && len(merges) <= 1<<24 {
		t.compactRanks = make([]byte, 0, unique*6)
	} else {
		t.ranks = make([]rankEntry, 0, unique)
	}
	group, count := 0, 0
	for i, entry := range entries {
		if i > 0 && entry.left == entries[i-1].left && entry.right == entries[i-1].right {
			continue
		}
		for group < int(entry.left) {
			group++
			if t.shortRanks != nil {
				setOffset24(t.shortRanks, group, uint32(count))
			} else {
				t.rankOffsets[group] = uint32(count)
			}
		}
		if t.ranks != nil {
			t.ranks = append(t.ranks, rankEntry{entry.right, entry.rank})
		} else {
			t.compactRanks = append(t.compactRanks, byte(entry.right), byte(entry.right>>8), byte(entry.right>>16),
				byte(entry.rank), byte(entry.rank>>8), byte(entry.rank>>16))
		}
		count++
	}
	for group < len(tokens) {
		group++
		if t.shortRanks != nil {
			setOffset24(t.shortRanks, group, uint32(count))
		} else {
			t.rankOffsets[group] = uint32(count)
		}
	}
	return t, nil
}

// skipSpace skips JSON whitespace.
func skipSpace(b []byte, pos int) int {
	for pos < len(b) && (b[pos] == ' ' || b[pos] == '\t' || b[pos] == '\n' || b[pos] == '\r') {
		pos++
	}
	return pos
}

// scanJSONString decodes one JSON string at b[pos] (opening quote), handling
// the standard escapes including \u pairs, and returns the string and the
// position after the closing quote.
func scanJSONString(b []byte, pos int) (string, int, error) {
	if pos >= len(b) || b[pos] != '"' {
		return "", pos, fmt.Errorf("expected string")
	}
	pos++
	var out []byte
	for {
		if pos >= len(b) {
			return "", pos, fmt.Errorf("unterminated string")
		}
		c := b[pos]
		if c == '"' {
			return string(out), pos + 1, nil
		}
		if c != '\\' {
			out = append(out, c)
			pos++
			continue
		}
		pos++
		if pos >= len(b) {
			return "", pos, fmt.Errorf("unterminated escape")
		}
		switch e := b[pos]; e {
		case '"', '\\', '/':
			out = append(out, e)
		case 'b':
			out = append(out, '\b')
		case 'f':
			out = append(out, '\f')
		case 'n':
			out = append(out, '\n')
		case 'r':
			out = append(out, '\r')
		case 't':
			out = append(out, '\t')
		case 'u':
			if pos+4 >= len(b) {
				return "", pos, fmt.Errorf("bad \\u escape")
			}
			var r rune
			for _, h := range b[pos+1 : pos+5] {
				var v byte
				switch {
				case h >= '0' && h <= '9':
					v = h - '0'
				case h >= 'a' && h <= 'f':
					v = h - 'a' + 10
				case h >= 'A' && h <= 'F':
					v = h - 'A' + 10
				default:
					return "", pos, fmt.Errorf("bad \\u escape")
				}
				r = r*16 + rune(v)
			}
			pos += 4
			if r >= 0xD800 && r <= 0xDBFF {
				// High surrogate: expect a low surrogate pair.
				if pos+5 >= len(b) || b[pos+1] != '\\' || b[pos+2] != 'u' {
					return "", pos, fmt.Errorf("lone surrogate")
				}
				var lo rune
				for _, h := range b[pos+3 : pos+7] {
					var v byte
					switch {
					case h >= '0' && h <= '9':
						v = h - '0'
					case h >= 'a' && h <= 'f':
						v = h - 'a' + 10
					case h >= 'A' && h <= 'F':
						v = h - 'A' + 10
					default:
						return "", pos, fmt.Errorf("bad surrogate")
					}
					lo = lo*16 + rune(v)
				}
				if lo < 0xDC00 || lo > 0xDFFF {
					return "", pos, fmt.Errorf("lone surrogate")
				}
				r = 0x10000 + (r-0xD800)*0x400 + (lo - 0xDC00)
				pos += 6
			}
			var buf [utf8.UTFMax]byte
			out = append(out, buf[:utf8.EncodeRune(buf[:], r)]...)
		default:
			return "", pos, fmt.Errorf("bad escape")
		}
		pos++
	}
}

// scanJSONInt decodes one JSON integer at b[pos] (after leading whitespace)
// and returns its value and the position after it. Values outside the int
// range or with non-integer syntax are rejected.
func scanJSONInt(b []byte, pos int) (int, int, error) {
	pos = skipSpace(b, pos)
	start := pos
	if pos < len(b) && b[pos] == '-' {
		pos++
	}
	digits := pos
	for pos < len(b) && b[pos] >= '0' && b[pos] <= '9' {
		pos++
	}
	if digits == pos || (pos < len(b) && (b[pos] == '.' || b[pos] == 'e' || b[pos] == 'E')) {
		return 0, start, fmt.Errorf("expected integer")
	}
	n, err := strconv.Atoi(string(b[start:pos]))
	if err != nil {
		return 0, start, fmt.Errorf("bad integer")
	}
	return n, pos, nil
}

// Load reads the vocabulary, merge list and token types from a pack directory.
// The vocabulary is decoded as a token stream directly into the dense token
// slice, so no transient vocabulary map is ever built.
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
	vocabPath := filepath.Join(dir, "vocab.json")
	vb, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	// Single pass over the raw bytes: vocab ids are contract-dense, so pairs
	// accumulate in a small id->token map (int keys only, no token copies as
	// map keys) and are densified once. encoding/json is not used here: its
	// token stream allocates ~25MB of scanner garbage on a 150k vocab.
	byID := make(map[int]string)
	maxID := -1
	pos := skipSpace(vb, 0)
	if pos >= len(vb) || vb[pos] != '{' {
		return nil, fmt.Errorf("vocab.json: expected object")
	}
	pos++
	pos = skipSpace(vb, pos)
	if pos < len(vb) && vb[pos] == '}' {
		pos++
	} else {
		for {
			var s string
			s, pos, err = scanJSONString(vb, pos)
			if err != nil {
				return nil, fmt.Errorf("vocab.json: %w", err)
			}
			pos = skipSpace(vb, pos)
			if pos >= len(vb) || vb[pos] != ':' {
				return nil, fmt.Errorf("vocab.json: expected ':'")
			}
			pos++
			var id int
			id, pos, err = scanJSONInt(vb, pos)
			if err != nil {
				return nil, fmt.Errorf("vocab.json: %w", err)
			}
			if id < 0 {
				return nil, fmt.Errorf("invalid vocabulary id %d", id)
			}
			if _, dup := byID[id]; dup {
				return nil, fmt.Errorf("invalid vocabulary id %d", id)
			}
			byID[id] = s
			if id > maxID {
				maxID = id
			}
			pos = skipSpace(vb, pos)
			if pos >= len(vb) {
				return nil, fmt.Errorf("vocab.json: unexpected end")
			}
			if vb[pos] == ',' {
				pos++
				continue
			}
			if vb[pos] == '}' {
				pos++
				break
			}
			return nil, fmt.Errorf("vocab.json: expected ',' or '}'")
		}
	}
	pos = skipSpace(vb, pos)
	if pos != len(vb) {
		return nil, fmt.Errorf("vocab.json: unexpected data after top-level value")
	}
	tokens := make([]string, maxID+1)
	if len(byID) != len(tokens) {
		return nil, fmt.Errorf("vocab.json: ids are not dense 0..%d", maxID)
	}
	for id, s := range byID {
		tokens[id] = s
	}
	byID = nil
	var types []int
	if err := read("token_type.json", &types); err != nil {
		return nil, err
	}
	mf, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	defer mf.Close()
	// Split on '\n' exactly like strings.Split(string(b), "\n"): keep '\r',
	// keep empty lines, and yield a trailing empty entry when the file ends
	// with a newline (or is empty), without ever materializing string(b).
	var merges []string
	var pending []byte
	br := bufio.NewReader(mf)
	for {
		line, rerr := br.ReadBytes('\n')
		if len(line) > 0 && line[len(line)-1] == '\n' {
			if len(pending) == 0 {
				merges = append(merges, string(line[:len(line)-1]))
			} else {
				pending = append(pending, line[:len(line)-1]...)
				merges = append(merges, string(pending))
				pending = nil
			}
			continue
		}
		pending = append(pending, line...)
		if rerr == io.EOF {
			merges = append(merges, string(pending))
			break
		}
		if rerr != nil {
			return nil, rerr
		}
	}
	tok, err := New(tokens, merges, types, split)
	if err != nil {
		return nil, err
	}
	// Transient vocab map, token aliases and file buffers are dead; hand pages back now.
	debug.FreeOSMemory()
	return tok, nil
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
		left, leftOK := t.tokenID(parts[i])
		right, rightOK := t.tokenID(parts[j])
		if leftOK && rightOK {
			if r, ok := t.mergeRank(left, right); ok {
				heap.Push(h, candidate{int(r), i, j, parts[i], parts[j]})
			}
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
		id, ok := t.tokenID(parts[i])
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
	if id < 0 || id >= t.VocabularySize() {
		return nil, fmt.Errorf("token id %d out of range", id)
	}
	s := t.token(id)
	if len(t.special) > 0 && t.special[id/8]&(1<<(id%8)) != 0 {
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
