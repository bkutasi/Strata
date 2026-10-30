package server

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"strings"
	"testing"
	"time"

	"strata/engine"
	"strata/outputparser"
	"strata/requestnorm"
)

func TestRequestStatusTracksProgressAndPhases(t *testing.T) {
	var output bytes.Buffer
	start := time.Unix(100, 0)
	status := newRequestStatus(log.New(&output, "", 0), 100, 60, "reasoning", start)
	status.prompt(start, 25, 50)
	if status.current != 25 || status.total != 50 || !status.promptShown || output.Len() == 0 {
		t.Fatalf("prompt status: %#v", status)
	}
	for _, tc := range []struct {
		event outputparser.Event
		phase string
		tool  string
	}{
		{outputparser.Event{Kind: "reasoning", Text: "private-output"}, "thinking", ""},
		{outputparser.Event{Kind: "content", Text: "private-output"}, "answering", ""},
		{outputparser.Event{Kind: "tool_start", Call: &outputparser.ToolCall{Name: "lookup"}}, "writing a tool call", "lookup"},
		{outputparser.Event{Kind: "tool_call"}, "tool call complete", ""},
	} {
		start = start.Add(time.Second)
		before := output.Len()
		status.note(start, 3, []outputparser.Event{tc.event})
		if status.phase != tc.phase || status.tool != tc.tool || status.count != 3 || output.Len() <= before {
			t.Fatalf("status after %s: %#v", tc.event.Kind, status)
		}
	}
	before := output.Len()
	status.done(start.Add(time.Second), generationResult{
		count:  3,
		result: engine.Result{Generated: 3, DecodeMS: 100, Finish: "stop", Extra: []string{"0", "0", "0", "9", "10"}},
	}, nil, nil, false, 256)
	if output.Len() <= before || strings.Contains(output.String(), "private-output") {
		t.Fatal("completion not written or generated text leaked into the terminal log")
	}
}

func TestRequestStatusNilLogger(t *testing.T) {
	if got := newRequestStatus(nil, 10, 0, "", time.Unix(0, 0)); got != nil {
		t.Fatalf("disabled logger created status: %#v", got)
	}
}

func TestSummarizeRequestMetrics(t *testing.T) {
	out := generationResult{count: 3, result: engine.Result{
		Generated: 3, DecodeMS: 100, Finish: "stop",
		Extra: []string{"0", "0", "0", "9", "10"},
	}}
	got := summarizeRequest(out, nil, nil)
	if got.finish != "stop" || got.canceled || !got.hasSpeed || got.speed != 30 || !got.hasCache || got.cache != 90 {
		t.Fatalf("summary = %#v", got)
	}
	got = summarizeRequest(out, context.Canceled, context.Canceled)
	if got.finish != "cancel" || !got.canceled {
		t.Fatalf("canceled summary = %#v", got)
	}
	got = summarizeRequest(out, errors.New("engine failed"), nil)
	if got.finish != "error" || got.canceled {
		t.Fatalf("error summary = %#v", got)
	}
	out.result.Generated, out.result.Extra = 1, []string{"0", "0", "0", "1", "0"}
	got = summarizeRequest(out, nil, nil)
	if got.hasSpeed || got.hasCache {
		t.Fatalf("summary with unavailable metrics = %#v", got)
	}
}

func TestReasoningBudgetLogsOnlyAdjustments(t *testing.T) {
	for _, tc := range []struct {
		name    string
		max     int
		request int
		want    int
		wantLog bool
	}{
		{"automatic", 4096, -1, 3072, true},
		{"within budget", 4096, 200, 200, false},
		{"capped", 4096, 4000, 3072, true},
		{"large automatic", 65536, -1, 49152, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var output bytes.Buffer
			s := &server{cfg: Config{Family: requestnorm.FamilyQwen, StatusLogger: log.New(&output, "", 0)}}
			got := s.effectiveReasoningBudget(true, tc.max, tc.request)
			if got != tc.want || (output.Len() > 0) != tc.wantLog {
				t.Fatalf("budget=%d log=%t; want budget=%d log=%t", got, output.Len() > 0, tc.want, tc.wantLog)
			}
		})
	}
}

func BenchmarkRequestStatus(b *testing.B) {
	logger := log.New(io.Discard, "", 0)
	start := time.Unix(100, 0)
	events := []outputparser.Event{{Kind: "content"}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		status := newRequestStatus(logger, 100, 0, "", start)
		status.prompt(start, 1, 2)
		status.note(start.Add(time.Second), 1, events)
		status.done(start.Add(2*time.Second), generationResult{count: 1, result: engine.Result{Finish: "stop"}}, nil, nil, false, 256)
	}
}
