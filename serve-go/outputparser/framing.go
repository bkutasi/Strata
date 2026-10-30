package outputparser

import (
	"bytes"
	"unicode/utf8"

	"strata/toolparser"
)

// workCounts is an optional, package-private test probe. The counters count
// bytes copied into buffered text and bytes examined by the resumable scanners.
type workCounts struct {
	buffer, frame, argument int
	openerCopy, openerScan  int
	pendingCopy             int
	heldStringCopy          int
}

func appendTracked(buf []byte, text string, work *int) []byte {
	if work != nil {
		*work += len(text)
		if len(text) > cap(buf)-len(buf) {
			*work += len(buf)
		}
	}
	return append(buf, text...)
}

func count(work *int, n int) {
	if work != nil {
		*work += n
	}
}

// needleSearch retains only a tag-sized overlap when a match is incomplete.
type needleSearch struct {
	pos   int
	found int
}

func newNeedle(start int) needleSearch { return needleSearch{pos: start, found: -1} }

func (s *needleSearch) find(data []byte, needle string, work *int) int {
	if s.found >= 0 {
		return s.found
	}
	part := data[s.pos:]
	i := bytes.Index(part, []byte(needle))
	if i >= 0 {
		count(work, i+len(needle))
		s.found = s.pos + i
		return s.found
	}
	count(work, len(part))
	s.pos = max(s.pos, len(data)-len(needle)+1)
	return -1
}

// paramFinder finds the first </parameter> followed by whitespace and either
// <parameter= or </function>. An undecided follower is resumed in place.
type paramFinder struct {
	search    needleSearch
	candidate int
	follower  int
	first     int
	done      int
}

func newParamFinder(start int) paramFinder {
	return paramFinder{search: newNeedle(start), candidate: -1, first: -1, done: -1}
}

func skipSpace(data []byte, pos int, work *int) int {
	for pos < len(data) {
		r, n := utf8.DecodeRune(data[pos:])
		count(work, n)
		if !space(r) {
			break
		}
		pos += n
	}
	return pos
}

// prefix reports a complete match (1), a possible incomplete match (0), or a
// mismatch (-1). Tags are short, so rechecking an incomplete prefix is bounded.
func prefix(data []byte, tag string, work *int) int {
	n := min(len(data), len(tag))
	count(work, n)
	if !bytes.Equal(data[:n], []byte(tag[:n])) {
		return -1
	}
	if len(data) < len(tag) {
		return 0
	}
	return 1
}

func (s *paramFinder) find(data []byte, work *int) int {
	if s.done >= 0 {
		return s.done
	}
	for {
		if s.candidate < 0 {
			at := s.search.find(data, toolparser.ParamEnd, work)
			if at < 0 {
				return -1
			}
			s.candidate = at
			if s.first < 0 {
				s.first = at
			}
			s.follower = at + len(toolparser.ParamEnd)
		}
		s.follower = skipSpace(data, s.follower, work)
		after := data[s.follower:]
		param := prefix(after, toolparser.ParamStart, work)
		function := prefix(after, toolparser.FuncEnd, work)
		if param == 1 || function == 1 {
			s.done = s.candidate
			return s.done
		}
		if param == 0 || function == 0 {
			return -2
		}
		// Between the candidate and follower there is only whitespace. No
		// other candidate can start there, so it need not be rescanned.
		s.search = newNeedle(s.follower)
		s.candidate = -1
	}
}

// frameScanner implements CallEnd and FunctionEnd together, without repeating
// either structural walk after another chunk. All offsets are byte offsets.
type frameScanner struct {
	state       string
	pos         int
	end         int
	functionEnd int
	param       paramFinder
	fallback    needleSearch
	next        nextCallScanner
}

func newFrameScanner() frameScanner {
	return frameScanner{state: "start", end: -1, functionEnd: -1}
}

func (s *frameScanner) process(data []byte, work *int) {
	for s.end < 0 {
		switch s.state {
		case "start":
			s.pos = skipSpace(data, s.pos, work)
			switch prefix(data[s.pos:], toolparser.FuncStart, work) {
			case 0:
				return
			case -1:
				s.fallback = newNeedle(s.pos)
				s.state = "fallback"
			case 1:
				s.pos += len(toolparser.FuncStart)
				s.state = "name"
			}
		case "name", "param_name":
			i := bytes.IndexByte(data[s.pos:], '>')
			if i < 0 {
				count(work, len(data)-s.pos)
				s.pos = len(data)
				return
			}
			count(work, i+1)
			s.pos += i + 1
			if s.state == "name" {
				s.state = "between"
			} else {
				s.param = newParamFinder(s.pos)
				s.state = "param"
			}
		case "between":
			s.pos = skipSpace(data, s.pos, work)
			rest := data[s.pos:]
			param := prefix(rest, toolparser.ParamStart, work)
			function := prefix(rest, toolparser.FuncEnd, work)
			if param == 1 {
				s.pos += len(toolparser.ParamStart)
				s.state = "param_name"
			} else if function == 1 {
				s.pos += len(toolparser.FuncEnd)
				s.functionEnd = s.pos
				s.next = newNextCallScanner(s.pos)
				s.state = "after_function"
			} else if param == 0 || function == 0 {
				return
			} else {
				s.fallback = newNeedle(s.pos)
				s.state = "fallback"
			}
		case "param":
			end := s.param.find(data, work)
			if end < 0 {
				return
			}
			s.pos = end + len(toolparser.ParamEnd)
			s.state = "between"
		case "after_function":
			s.pos = skipSpace(data, s.pos, work)
			switch prefix(data[s.pos:], callEnd, work) {
			case 0:
				return
			case 1:
				s.end = s.pos
			case -1:
				s.fallback = newNeedle(s.pos)
				s.state = "fallback"
			}
		case "fallback":
			s.end = s.fallback.find(data, callEnd, work)
			return
		}
	}
}

// nextCallScanner recognizes the next complete wrapper opener after a function
// without rescanning whitespace or a long function name on each feed.
type nextCallScanner struct {
	state string
	pos   int
	done  bool
	fail  bool
}

func newNextCallScanner(start int) nextCallScanner {
	return nextCallScanner{state: "space", pos: start}
}

func (s *nextCallScanner) process(data []byte, work *int) bool {
	for !s.done && !s.fail {
		switch s.state {
		case "space", "call_space", "name_space", "after_name":
			s.pos = skipSpace(data, s.pos, work)
			if s.pos == len(data) {
				return false
			}
			switch s.state {
			case "space":
				s.state = "call"
			case "call_space":
				s.state = "function"
			case "name_space":
				s.state = "name"
			case "after_name":
				count(work, 1)
				if data[s.pos] == '>' {
					s.done = true
				} else {
					s.fail = true
				}
			}
		case "call", "function":
			tag := callStart
			if s.state == "function" {
				tag = toolparser.FuncStart
			}
			switch prefix(data[s.pos:], tag, work) {
			case 0:
				return false
			case -1:
				s.fail = true
			case 1:
				s.pos += len(tag)
				if s.state == "call" {
					s.state = "call_space"
				} else {
					s.state = "name_space"
				}
			}
		case "name", "name_more":
			if s.pos == len(data) {
				return false
			}
			r, n := utf8.DecodeRune(data[s.pos:])
			count(work, n)
			if r == '<' || r == '>' || space(r) {
				if s.state == "name" {
					s.fail = true
				} else {
					s.state = "after_name"
				}
			} else {
				s.pos += n
				s.state = "name_more"
			}
		}
	}
	return s.done
}
