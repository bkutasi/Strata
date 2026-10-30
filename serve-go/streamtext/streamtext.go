// Package streamtext provides independent token decoding and content stop matching.
// Feed Decoder output to the caller's parser, then feed only content events to
// StopMatcher; call Boundary before emitting any non-content event.
package streamtext

import (
	"strings"
	"unicode/utf8"
)

// Decoder incrementally decodes token bytes. Its zero value is ready to use.
// Like the server's Detokenizer, an incomplete tail is not emitted at finish.
// Discard the decoder at the end of generation. It is not safe for concurrent use.
type Decoder struct {
	pending []byte
}

// StopMatcher cuts content at the first stop. It is not safe for concurrent use.
type StopMatcher struct {
	stops    []string
	held     string
	hit      string
	finished bool
}

// NewStopMatcher constructs a matcher. Empty stops are ignored, as in stop_strings.
// Stop strings must be valid UTF-8. Ties are resolved in input order.
func NewStopMatcher(stops []string) *StopMatcher {
	s := &StopMatcher{}
	for _, stop := range stops {
		if stop != "" {
			if !utf8.ValidString(stop) {
				panic("streamtext: stop string is not valid UTF-8")
			}
			s.stops = append(s.stops, stop)
		}
	}
	return s
}

// Hit returns the matched stop string, or "" if no stop has matched.
func (s *StopMatcher) Hit() string { return s.hit }

// Pending reports whether an incomplete UTF-8 character is buffered.
func (s *Decoder) Pending() bool { return len(s.pending) != 0 }

// Boundary releases a held stop prefix before a non-content parser event.
// It permits subsequent Push calls.
// Call it before delivering each tool or reasoning event; stops cannot span
// that event. It returns "" after a match or Finish.
func (s *StopMatcher) Boundary() string {
	if s.finished || s.hit != "" {
		return ""
	}
	held := s.held
	s.held = ""
	return held
}

// Finish releases an unmatched stop prefix and closes the matcher. It is idempotent.
// Process parser finish events before calling it; do not finalize the decoder.
func (s *StopMatcher) Finish() string {
	if s.finished {
		return ""
	}
	out := s.Boundary()
	s.finished = true
	return out
}

// Push consumes one token's raw bytes and returns complete decoded characters.
// Invalid sequences are replaced as in Python's incremental UTF-8 decoder.
func (s *Decoder) Push(token []byte) string {
	b := append(s.pending, token...)
	s.pending = nil
	var out strings.Builder
	for len(b) > 0 {
		// CPython delays rejecting a surrogate prefix until a third byte
		// arrives, unlike utf8.FullRune.
		surrogatePrefix := len(b) == 2 && b[0] == 0xed && b[1] >= 0xa0 && b[1] <= 0xbf
		if !utf8.FullRune(b) || surrogatePrefix {
			s.pending = append(s.pending, b...)
			break
		}
		r, n := utf8.DecodeRune(b)
		if r == utf8.RuneError && n == 1 {
			// Python replaces a valid but incomplete prefix as one unit, even
			// when the following byte invalidates that prefix.
			n = invalidPrefixLen(b)
			out.WriteRune(utf8.RuneError)
		} else {
			out.Write(b[:n])
		}
		b = b[n:]
	}
	return out.String()
}

func invalidPrefixLen(b []byte) int {
	first := b[0]
	want := 0
	switch {
	case first >= 0xc2 && first <= 0xdf:
		want = 2
	case first >= 0xe0 && first <= 0xef:
		want = 3
	case first >= 0xf0 && first <= 0xf4:
		want = 4
	default:
		return 1
	}
	n := 1
	for n < len(b) && n < want {
		c := b[n]
		if c < 0x80 || c > 0xbf || (n == 1 && ((first == 0xe0 && c < 0xa0) ||
			(first == 0xed && c > 0x9f) || (first == 0xf0 && c < 0x90) ||
			(first == 0xf4 && c > 0x8f))) {
			break
		}
		n++
	}
	return n
}

// Push consumes decoded content from the parser, not reasoning or tool events.
// Text must be valid UTF-8. It returns the content safe to emit.
// After a match or Finish, further calls return "".
func (s *StopMatcher) Push(text string) string {
	if s.finished || s.hit != "" {
		return ""
	}
	if !utf8.ValidString(text) {
		panic("streamtext: content is not valid UTF-8")
	}
	buf := s.held + text
	first := -1
	for _, stop := range s.stops {
		if i := strings.Index(buf, stop); i >= 0 && (first < 0 || i < first) {
			first, s.hit = i, stop
		}
	}
	if first >= 0 {
		s.held = ""
		return buf[:first]
	}
	keep := 0
	for _, stop := range s.stops {
		// Only rune boundaries can be prefixes of Python strings.
		for i := range stop {
			if i > keep && i <= len(buf) && strings.HasSuffix(buf, stop[:i]) {
				keep = i
			}
		}
	}
	s.held = buf[len(buf)-keep:]
	return buf[:len(buf)-keep]
}
