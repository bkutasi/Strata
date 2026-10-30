package outputparser

import (
	"unicode/utf8"

	"strata/toolparser"
)

// openerHold keeps a partial standard function opener without reconstructing
// or re-matching the entire buffered name on every feed. pos is relative to
// the bytes after "<function="; phase mirrors the functionPart grammar.
type openerHold struct {
	buf   []byte
	start int
	pos   int
	phase string
}

func (p *Parser) openerCopyWork() *int {
	if p.work == nil {
		return nil
	}
	return &p.work.openerCopy
}

func (p *Parser) openerScanWork() *int {
	if p.work == nil {
		return nil
	}
	return &p.work.openerScan
}

// holdOpener is called only after the normal content parser has classified
// the current buffer as a possible standard <function= opener. A fixed-size
// partial prefix or whitespace before the function tag can also be held.
func (p *Parser) holdOpener(i int) {
	if i < 0 {
		return
	}
	h := &openerHold{start: i + len(callStart), phase: "wrapper_space"}
	h.buf = appendTracked(h.buf, p.buf, p.openerCopyWork())
	if h.advance(p.openerScanWork()) {
		return
	}
	p.opener, p.buf = h, ""
}

// advance returns true when the opener is complete or can no longer match.
func (h *openerHold) advance(work *int) bool {
	if h.phase == "wrapper_space" {
		h.start = skipSpace(h.buf, h.start, work)
		if h.start == len(h.buf) {
			return false
		}
		h.phase = "function_prefix"
	}
	if h.phase == "function_prefix" {
		switch prefix(h.buf[h.start:], toolparser.FuncStart, work) {
		case 0:
			return false
		case -1:
			// Loose and non-function forms still use the normal parser.
			return true
		case 1:
			h.start += len(toolparser.FuncStart)
			h.phase = "leading"
		}
	}
	for h.start+h.pos < len(h.buf) {
		at := h.start + h.pos
		r, n := utf8.DecodeRune(h.buf[at:])
		count(work, n)
		h.pos += n
		if r == '<' || r == '>' {
			return true
		}
		if space(r) {
			if h.phase == "name" {
				h.phase = "trailing"
			}
		} else {
			if h.phase == "trailing" {
				return true
			}
			h.phase = "name"
		}
	}
	return false
}
