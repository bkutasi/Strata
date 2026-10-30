package server

import (
	"log"
	"strconv"
	"time"
	"unicode"

	"strata/outputparser"
)

// requestStatus holds only the current request's progress; it does not retain text.
type requestStatus struct {
	logger                *log.Logger
	started, first, last  time.Time
	current, total, count int
	max, budget           int
	phase, tool           string
	promptShown           bool
}

type requestSummary struct {
	finish   string
	canceled bool
	speed    float64
	hasSpeed bool
	cache    float64
	hasCache bool
}

func summarizeRequest(out generationResult, err, contextErr error) requestSummary {
	summary := requestSummary{finish: out.result.Finish, canceled: contextErr != nil}
	if out.stopped {
		summary.finish = "stop"
	}
	if summary.canceled {
		summary.finish = "cancel"
	} else if err != nil || summary.finish == "" {
		summary.finish = "error"
	}
	if out.result.Generated >= 2 && out.result.DecodeMS > 0 {
		summary.speed = float64(out.result.Generated) * 1000 / out.result.DecodeMS
		summary.hasSpeed = true
	}
	if len(out.result.Extra) >= 5 {
		hits, e1 := strconv.ParseInt(out.result.Extra[3], 10, 64)
		lookups, e2 := strconv.ParseInt(out.result.Extra[4], 10, 64)
		if e1 == nil && e2 == nil && hits >= 0 && lookups > 0 && hits <= lookups {
			summary.cache = float64(hits) * 100 / float64(lookups)
			summary.hasCache = true
		}
	}
	return summary
}

func newRequestStatus(logger *log.Logger, max, budget int, state string, now time.Time) *requestStatus {
	if logger == nil {
		return nil
	}
	phase := "answering"
	if state == "reasoning" {
		phase = "thinking"
	} else if state == "call" || state == "rcall" {
		phase = "writing a tool call"
	}
	s := &requestStatus{logger: logger, started: now, last: now, max: max, budget: budget, phase: phase}
	return s
}

func (s *requestStatus) prompt(now time.Time, current, total int) {
	s.current, s.total = current, total
	if !s.promptShown || now.Sub(s.last) >= time.Second {
		s.tick(now)
	}
}

func (s *requestStatus) note(now time.Time, count int, events []outputparser.Event) {
	s.count = count
	if s.first.IsZero() {
		s.first = now
	}
	for _, event := range events {
		switch event.Kind {
		case "reasoning":
			s.phase, s.tool = "thinking", ""
		case "content":
			s.phase, s.tool = "answering", ""
		case "tool_start":
			s.phase, s.tool = "writing a tool call", ""
			if event.Call != nil {
				name := event.Call.Name
				for _, r := range name {
					if unicode.IsControl(r) {
						name = ""
						break
					}
				}
				s.tool = name
			}
		case "tool_call":
			s.phase, s.tool = "tool call complete", ""
		}
	}
	if now.Sub(s.last) >= time.Second {
		s.tick(now)
	}
}

func (s *requestStatus) tick(now time.Time) {
	s.last = now
	elapsed := now.Sub(s.started).Seconds()
	if s.first.IsZero() {
		s.promptShown = true
		if s.total > 0 {
			s.logger.Printf("[strata] reading the prompt: %d of %d tokens, %.0f s so far", s.current, s.total, elapsed)
		} else {
			s.logger.Printf("[strata] reading the prompt: %.0f s so far", elapsed)
		}
		return
	}
	span := now.Sub(s.first).Seconds()
	if span < 0.25 {
		span = 0.25
	}
	phase := s.phase
	if s.tool != "" {
		phase += ": " + s.tool
	}
	if s.phase == "thinking" && s.budget > 0 {
		s.logger.Printf("[strata] %s: %d of budget %d (max %d) tokens, %.1f tok/s, %.0f s", phase, s.count, s.budget, s.max, float64(s.count)/span, elapsed)
	} else {
		s.logger.Printf("[strata] %s: %d of max %d tokens, %.1f tok/s, %.0f s", phase, s.count, s.max, float64(s.count)/span, elapsed)
	}
}

func (s *requestStatus) done(now time.Time, out generationResult, err, contextErr error, repeated bool, repeatLimit int) {
	if repeated {
		s.logger.Printf("[strata] the reply repeated one token %d times in a row and was ended as length (repeat_stop_tokens in the config; 0 turns this off)", repeatLimit)
	}
	summary := summarizeRequest(out, err, contextErr)
	cache := ""
	if summary.hasCache {
		cache = ", expert cache " + strconv.FormatFloat(summary.cache, 'f', 1, 64) + "% hit"
	}
	if summary.hasSpeed {
		s.logger.Printf("[strata] done: %d tokens in %.0f s (%.1f tok/s) (%s, cancel=%t)%s",
			out.count, now.Sub(s.started).Seconds(), summary.speed, summary.finish, summary.canceled, cache)
	} else {
		s.logger.Printf("[strata] done: %d tokens in %.0f s (no speed) (%s, cancel=%t)%s",
			out.count, now.Sub(s.started).Seconds(), summary.finish, summary.canceled, cache)
	}
	if summary.finish == "length" && s.phase == "thinking" {
		s.logger.Printf("[strata] the reply ended while still thinking, so it has no answer")
	}
}
