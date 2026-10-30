package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestStartNarratorKindsAndClockHeartbeat(t *testing.T) {
	now := time.Unix(100, 0)
	var events []narrationEvent
	n := newStartNarrator(func() time.Time { return now }, func(e narrationEvent) { events = append(events, e) })
	n.line("expert arena: loading")
	n.line("expert arena: loading")
	if len(events) != 1 || events[0].kind != "arena" || events[0].data != "expert arena: loading" {
		t.Fatalf("arena events = %#v", events)
	}
	now = now.Add(20 * time.Second)
	n.heartbeat()
	if len(events) != 2 || events[1].kind != "heartbeat" || events[1].data != "20" {
		t.Fatalf("heartbeat events = %#v", events)
	}
	n.line("strata: session is up")
	if len(events) != 3 || events[2].kind != "up" || events[2].data != "strata: session is up" {
		t.Fatalf("ready events = %#v", events)
	}
}

func TestNarrationLinesSplitAcrossChunks(t *testing.T) {
	var got []string
	lines := narrationLines{pending: make([]byte, 0, narrationMaxLineSize)}
	lines.feed([]byte("expert arena:"), func(s string) { got = append(got, s) }, func() { t.Fatal("unexpected oversized line") })
	if len(lines.pending) != len("expert arena:") {
		t.Fatalf("retained bytes = %d", len(lines.pending))
	}
	lines.feed([]byte(" loading\nnext\n"), func(s string) { got = append(got, s) }, func() { t.Fatal("unexpected oversized line") })
	if len(got) != 2 || got[0] != "expert arena: loading" || got[1] != "next" {
		t.Fatalf("lines = %#v", got)
	}
}

func TestNarrationLinesDiscardOversizedAndRecover(t *testing.T) {
	var got []string
	var errors int
	lines := narrationLines{pending: make([]byte, 0, narrationMaxLineSize)}
	lines.feed(bytes.Repeat([]byte("x"), narrationMaxLineSize), func(s string) { got = append(got, s) }, func() { errors++ })
	if len(lines.pending) != narrationMaxLineSize || lines.drop {
		t.Fatalf("state at exact bound: retained=%d dropping=%v", len(lines.pending), lines.drop)
	}
	lines.feed([]byte("y\nrecovered\n"), func(s string) { got = append(got, s) }, func() { errors++ })
	if errors != 1 || len(got) != 1 || got[0] != "recovered" {
		t.Fatalf("oversized recovery: errors=%d lines=%#v", errors, got)
	}
	if cap(lines.pending) > narrationMaxLineSize || len(lines.pending) > narrationMaxLineSize {
		t.Fatalf("retained storage exceeds bound: len=%d cap=%d", len(lines.pending), cap(lines.pending))
	}
	if strings.Contains(strings.Join(got, ""), "x") {
		t.Fatal("discarded oversized line was emitted")
	}
}
