// Package tokenizer implements Strata's Unicode pre-tokenizer splits.
// It does not perform BPE, recognize special tokens, or load a vocabulary.
package tokenizer

import (
	"errors"
	"unicode/utf8"
)

// Span is a half-open range of UTF-8 byte offsets into the input string.
type Span struct {
	Start int
	End   int
}

// ErrInvalidUTF8 reports an input that cannot correspond to a Python Unicode
// string encoded as UTF-8.
var ErrInvalidUTF8 = errors.New("tokenizer: input is not valid UTF-8")

type character struct {
	r     rune
	class byte
	start int
}

type piece struct {
	start int // character index
	end   int
}

type scanner struct {
	text  string
	chars []character
}

func newScanner(text string) (scanner, error) {
	if !utf8.ValidString(text) {
		return scanner{}, ErrInvalidUTF8
	}
	s := scanner{text: text}
	for start, r := range text {
		s.chars = append(s.chars, character{r: r, class: runeClass(r), start: start})
	}
	return s, nil
}

func (s scanner) spans(pieces []piece) []Span {
	out := make([]Span, 0, len(pieces))
	for _, p := range pieces {
		end := len(s.text)
		if p.end < len(s.chars) {
			end = s.chars[p.end].start
		}
		out = append(out, Span{Start: s.chars[p.start].start, End: end})
	}
	return out
}

// Qwen35 applies the ordered Qwen 3.5 pre-tokenizer pattern. The result is
// ordered, covers the input exactly, and contains no empty spans. Empty input
// returns an empty slice. Invalid UTF-8 returns ErrInvalidUTF8.
func Qwen35(text string) ([]Span, error) {
	s, err := newScanner(text)
	if err != nil {
		return nil, err
	}
	pieces := s.split([]piece{{0, len(s.chars)}}, s.qwen)
	return s.spans(pieces), nil
}

// DeepSeek3 applies the DeepSeek V3/V4.1 patterns sequentially: number runs,
// CJK runs, then the general pattern. Each pass splits every previous piece,
// retaining the unmatched regions as independent pieces.
// Its spans and invalid-input behavior are the same as Qwen35's.
func DeepSeek3(text string) ([]Span, error) {
	s, err := newScanner(text)
	if err != nil {
		return nil, err
	}
	pieces := []piece{{0, len(s.chars)}}
	pieces = s.split(pieces, s.number)
	pieces = s.split(pieces, s.cjk)
	pieces = s.split(pieces, s.deepSeek)
	return s.spans(pieces), nil
}

// split implements finditer and retains both matches and unmatched stretches.
// A matcher returns its exclusive end, or i if it does not match at i.
func (s scanner) split(pieces []piece, match func(int, int) int) []piece {
	var out []piece
	for _, p := range pieces {
		start := p.start
		for i := p.start; i < p.end; {
			end := match(i, p.end)
			if end == i {
				i++
				continue
			}
			if start < i {
				out = append(out, piece{start, i})
			}
			out = append(out, piece{i, end})
			i, start = end, end
		}
		if start < p.end {
			out = append(out, piece{start, p.end})
		}
	}
	return out
}

func (s scanner) has(i int, mask byte) bool {
	return s.chars[i].class&mask != 0
}

func lineBreak(r rune) bool { return r == '\r' || r == '\n' }

func (s scanner) qwen(i, end int) int {
	// The contraction alternative precedes the general letter alternative.
	if s.chars[i].r == '\'' && i+1 < end {
		switch s.chars[i+1].r {
		case 's', 'S', 't', 'T', 'm', 'M', 'd', 'D':
			return i + 2
		case 'r', 'R', 'v', 'V':
			if i+2 < end && (s.chars[i+2].r == 'e' || s.chars[i+2].r == 'E') {
				return i + 3
			}
		case 'l', 'L':
			if i+2 < end && (s.chars[i+2].r == 'l' || s.chars[i+2].r == 'L') {
				return i + 3
			}
		}
	}
	if j := s.word(i, end, classLetter|classNumber); j > i {
		return j
	}
	if s.has(i, classNumber) {
		return i + 1
	}
	j := i
	if s.chars[j].r == ' ' && j+1 < end {
		j++
	}
	start := j
	for j < end && !s.has(j, classSpace|classLetter|classMark|classNumber) {
		j++
	}
	if j > start {
		return s.trailingLineBreaks(j, end)
	}
	return s.whitespace(i, end)
}

func (s scanner) number(i, end int) int {
	j := i
	for j < end && j-i < 3 && s.has(j, classNumber) {
		j++
	}
	return j
}

func (s scanner) cjk(i, end int) int {
	j := i
	for j < end {
		r := s.chars[j].r
		if !(r >= 0x4e00 && r <= 0x9fa5 || r >= 0x3040 && r <= 0x309f || r >= 0x30a0 && r <= 0x30ff) {
			break
		}
		j++
	}
	return j
}

func (s scanner) deepSeek(i, end int) int {
	if asciiPunctuation(s.chars[i].r) && i+1 < end && asciiLetter(s.chars[i+1].r) {
		j := i + 2
		for j < end && asciiLetter(s.chars[j].r) {
			j++
		}
		return j
	}
	if j := s.word(i, end, classLetter|classPunctuation|classSymbol); j > i {
		return j
	}
	j := i
	if s.chars[j].r == ' ' && j+1 < end {
		j++
	}
	start := j
	for j < end && s.has(j, classPunctuation|classSymbol) {
		j++
	}
	if j > start {
		return s.trailingLineBreaks(j, end)
	}
	return s.whitespace(i, end)
}

// word matches [^\r\n<excluded>]?[\p{L}\p{M}]+. The optional
// character is greedy, but must leave at least one letter or mark.
func (s scanner) word(i, end int, excluded byte) int {
	j := i
	if !lineBreak(s.chars[i].r) && !s.has(i, excluded) &&
		i+1 < end && s.has(i+1, classLetter|classMark) {
		j++
	}
	if !s.has(j, classLetter|classMark) {
		return i
	}
	for j < end && s.has(j, classLetter|classMark) {
		j++
	}
	return j
}

func (s scanner) trailingLineBreaks(i, end int) int {
	for i < end && lineBreak(s.chars[i].r) {
		i++
	}
	return i
}

// whitespace implements the three ordered regex alternatives
// \s*[\r\n]+ | \s+(?!\S) | \s+ including their greedy backtracking.
func (s scanner) whitespace(i, end int) int {
	j, lastLineBreak := i, -1
	for j < end && s.has(j, classSpace) {
		if lineBreak(s.chars[j].r) {
			lastLineBreak = j
		}
		j++
	}
	if lastLineBreak >= i {
		return lastLineBreak + 1
	}
	if j == i {
		return i
	}
	if j < end && j-i > 1 {
		return j - 1 // leave one whitespace character for the negative lookahead
	}
	return j
}

func asciiPunctuation(r rune) bool {
	return r >= '!' && r <= '/' || r >= ':' && r <= '@' ||
		r >= '[' && r <= '`' || r >= '{' && r <= '~'
}

func asciiLetter(r rune) bool {
	return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z'
}
