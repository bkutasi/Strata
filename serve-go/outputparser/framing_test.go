package outputparser

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"strata/toolparser"
)

func feedPieces(t *testing.T, p *Parser, text string, width int) []Event {
	t.Helper()
	var all []Event
	for len(text) > 0 {
		n := min(width, len(text))
		events, err := p.Feed(text[:n])
		if err != nil {
			t.Fatalf("feed: %v", err)
		}
		all = append(all, events...)
		text = text[n:]
	}
	return all
}

func TestIncrementalStructuralWork(t *testing.T) {
	const repetitions = 4096
	for _, thinking := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, stringValue := range []bool{false, true} {
				name := fmt.Sprintf("thinking=%t/stream=%t/string=%t", thinking, stream, stringValue)
				t.Run(name, func(t *testing.T) {
					prop := map[string]any{"type": "integer"}
					if stringValue {
						prop["type"] = "string"
					}
					tools := []map[string]any{{"name": "f", "parameters": map[string]any{
						"properties": map[string]any{"x": prop},
					}}}
					p, err := New(Options{Thinking: thinking, Tools: tools, StreamTools: stream})
					if err != nil {
						t.Fatal(err)
					}
					counts := &workCounts{}
					p.work = counts
					opener := "<tool_call><function=f><parameter=x>"
					events := feedPieces(t, p, opener, 1)
					before := *counts
					var chunks strings.Builder
					// Include decoy closers, literal function closers and a
					// long undecided follower, all delivered one byte at a time.
					chunks.WriteString(strings.Repeat("x</parameter>!", repetitions/16))
					chunks.WriteString(strings.Repeat("q</tool_call>z", repetitions/16))
					chunks.WriteString("</parameter>")
					chunks.WriteString(strings.Repeat(" ", repetitions))
					middle := chunks.String()
					middleEvents := feedPieces(t, p, middle, 1)
					events = append(events, middleEvents...)
					for _, event := range middleEvents {
						if event.Kind == "tool_call" {
							t.Fatal("emitted a final call for an undecided parameter end")
						}
					}
					delta := workCounts{buffer: counts.buffer - before.buffer, frame: counts.frame - before.frame,
						argument: counts.argument - before.argument}
					// Fixed bounds count structural bytes examined and geometric
					// buffer copies, not elapsed time or runtime allocations.
					limit := 64 * (len(middle) + len(opener))
					if delta.buffer > limit || delta.frame > limit || delta.argument > limit {
						t.Fatalf("nonlinear work for %d input bytes: buffer=%d frame=%d argument=%d (limit %d)",
							len(middle), delta.buffer, delta.frame, delta.argument, limit)
					}
					if delta.frame == 0 || stream && !thinking && delta.argument == 0 {
						t.Fatalf("operation probe did not observe work: %#v", delta)
					}
					suffix := "</function></tool_call>"
					if thinking {
						suffix += "</think>"
					}
					events = append(events, feedPieces(t, p, suffix, 1)...)
					tail, err := p.Finish("stop")
					if err != nil {
						t.Fatal(err)
					}
					events = append(events, tail...)
					var calls int
					var call *ToolCall
					var streamed strings.Builder
					for _, event := range events {
						if event.Kind == "tool_call" {
							calls++
							call = event.Call
						} else if event.Kind == "tool_args" {
							streamed.WriteString(event.Text)
						}
					}
					if calls != 1 {
						t.Fatalf("got %d final calls", calls)
					}
					if stream && !thinking {
						var arguments map[string]any
						if err := json.Unmarshal([]byte(streamed.String()), &arguments); err != nil {
							t.Fatalf("streamed arguments: %v", err)
						}
						if arguments["x"] != call.Arguments["x"] {
							t.Fatal("streamed and final argument values differ")
						}
					}
				})
			}
		}
	}
}

func TestLongIncrementalOpeners(t *testing.T) {
	for _, stream := range []bool{false, true} {
		p, err := New(Options{StreamTools: stream})
		if err != nil {
			t.Fatal(err)
		}
		counts := &workCounts{}
		p.work = counts
		feedPieces(t, p, "<tool_call><function=f><parameter=", 1)
		before := *counts
		longName := strings.Repeat("x", 4096)
		feedPieces(t, p, longName, 1)
		feedPieces(t, p, ">v</parameter></function><tool_call><function=", 1)
		feedPieces(t, p, longName, 1)
		work := workCounts{buffer: counts.buffer - before.buffer, frame: counts.frame - before.frame,
			argument: counts.argument - before.argument}
		limit := 64 * (2*len(longName) + 100)
		if work.buffer > limit || work.frame > limit || work.argument > limit {
			t.Fatalf("stream=%t: non-linear opener work: %#v, limit %d", stream, work, limit)
		}
		events := feedPieces(t, p, "></function></tool_call>", 1)
		end, err := p.Finish("stop")
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, end...)
		calls := 0
		for _, event := range events {
			if event.Kind == "tool_call" {
				calls++
			}
		}
		if calls != 2 {
			t.Fatalf("stream=%t: expected two calls, got %d", stream, calls)
		}
	}
}

func TestFirstOpenerIncrementalWork(t *testing.T) {
	const size = 8192
	name := strings.Repeat("x", size)
	for _, declared := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, ending := range []string{"complete", "invalid", "unfinished"} {
				t.Run(fmt.Sprintf("declared=%t/stream=%t/%s", declared, stream, ending), func(t *testing.T) {
					var tools []map[string]any
					if declared {
						tools = []map[string]any{{"name": name}}
					}
					p, err := New(Options{Tools: tools, StreamTools: stream})
					if err != nil {
						t.Fatal(err)
					}
					counts := &workCounts{}
					p.work = counts
					feedPieces(t, p, "prefix\n<tool_call><function=", 1)
					before := *counts
					events := feedPieces(t, p, name, 1)
					if len(events) != 0 || p.Status().Clean || p.Status().State != "content" {
						t.Fatalf("partial name emitted events or lost held state: %#v %#v", events, p.Status())
					}
					copies := counts.openerCopy - before.openerCopy
					scans := counts.openerScan - before.openerScan
					heldCopies := counts.heldStringCopy - before.heldStringCopy
					if copies < size || copies > 16*size || scans != size || heldCopies != 0 {
						t.Fatalf("opener work for %d bytes: copied=%d scanned=%d held-copies=%d",
							size, copies, scans, heldCopies)
					}
					switch ending {
					case "complete":
						events = feedPieces(t, p, "></function></tool_call>", 1)
					case "invalid":
						events = feedPieces(t, p, " x", 1)
					}
					tail, err := p.Finish("stop")
					if err != nil {
						t.Fatal(err)
					}
					events = append(events, tail...)
					var content strings.Builder
					var calls []*ToolCall
					for _, event := range events {
						if event.Kind == "content" {
							content.WriteString(event.Text)
						} else if event.Kind == "tool_call" {
							calls = append(calls, event.Call)
						}
					}
					if ending == "complete" {
						if len(calls) != 1 || calls[0].Name != name {
							t.Fatalf("complete opener yielded %d calls", len(calls))
						}
					} else {
						want := "\n<tool_call><function=" + name
						if ending == "invalid" {
							want += " x"
						}
						if len(calls) != 0 || content.String() != want {
							t.Fatalf("%s opener: %d calls, content length %d want %d",
								ending, len(calls), content.Len(), len(want))
						}
					}
				})
			}
		}
	}
}

func TestFirstOpenerWrapperWhitespaceWork(t *testing.T) {
	const size = 8192
	for _, stream := range []bool{false, true} {
		p, err := New(Options{StreamTools: stream})
		if err != nil {
			t.Fatal(err)
		}
		counts := &workCounts{}
		p.work = counts
		feedPieces(t, p, "<tool_call>", 1)
		before := *counts
		events := feedPieces(t, p, strings.Repeat(" ", size), 1)
		copies, scans := counts.openerCopy-before.openerCopy, counts.openerScan-before.openerScan
		if len(events) != 0 || p.Status().Clean || copies < size || copies > 16*size ||
			scans != size || counts.heldStringCopy != before.heldStringCopy {
			t.Fatalf("wrapper whitespace: events=%#v copies=%d scans=%d status=%#v",
				events, copies, scans, p.Status())
		}
		events = feedPieces(t, p, "<function=f></function></tool_call>", 1)
		tail, err := p.Finish("stop")
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, tail...)
		calls := 0
		for _, event := range events {
			if event.Kind == "tool_call" {
				calls++
			}
		}
		if calls != 1 {
			t.Fatalf("expected one call, got %d", calls)
		}
	}
}

func TestPendingReasoningWhitespaceIncrementalWork(t *testing.T) {
	const size = 8192
	call := "<tool_call><function=f></function></tool_call>"
	for _, ending := range []string{"stop", "length", "prose", "think", "second"} {
		t.Run(ending, func(t *testing.T) {
			p, err := New(Options{Thinking: true, Tools: []map[string]any{{"name": "f"}}})
			if err != nil {
				t.Fatal(err)
			}
			counts := &workCounts{}
			p.work = counts
			events := feedPieces(t, p, call, 1)
			if len(events) != 0 || p.Status().Pending != 1 {
				t.Fatalf("call was not held: %#v %#v", events, p.Status())
			}
			before := counts.pendingCopy
			beforeHeld := counts.heldStringCopy
			events = feedPieces(t, p, strings.Repeat(" ", size), 1)
			copies := counts.pendingCopy - before
			heldCopies := counts.heldStringCopy - beforeHeld
			if len(events) != 0 || p.Status().Pending != 1 || copies < size || copies > 16*size || heldCopies != 0 {
				t.Fatalf("pending work: events=%#v pending=%d copies=%d held-copies=%d",
					events, p.Status().Pending, copies, heldCopies)
			}
			reason := "stop"
			switch ending {
			case "length":
				reason = "length"
			case "prose":
				events = feedPieces(t, p, "continued", 1)
			case "think":
				events = feedPieces(t, p, "</think>answer", 1)
			case "second":
				events = feedPieces(t, p, call+"</think>", 1)
			}
			tail, err := p.Finish(reason)
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, tail...)
			var reasoning strings.Builder
			calls := 0
			for _, event := range events {
				if event.Kind == "reasoning" {
					reasoning.WriteString(event.Text)
				} else if event.Kind == "tool_call" {
					calls++
				}
			}
			expectedCalls := 1
			if ending == "length" || ending == "prose" {
				expectedCalls = 0
			} else if ending == "second" {
				expectedCalls = 2
			}
			if calls != expectedCalls {
				t.Fatalf("got %d calls, want %d", calls, expectedCalls)
			}
			if ending == "length" || ending == "prose" {
				want := call + strings.Repeat(" ", size)
				if ending == "prose" {
					want += "continued"
				}
				if reasoning.String() != want {
					t.Fatalf("reasoning length %d, want %d", reasoning.Len(), len(want))
				}
			}
		})
	}
}

func TestProvisionalStreamedCallEndsUnfinished(t *testing.T) {
	for _, reason := range []string{"length", "error", "cancel"} {
		t.Run(reason, func(t *testing.T) {
			p, err := New(Options{StreamTools: true})
			if err != nil {
				t.Fatal(err)
			}
			events, err := p.Feed("<tool_call><function=f><parameter=x>unfinished")
			if err != nil {
				t.Fatal(err)
			}
			tail, err := p.Finish(reason)
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, tail...)
			starts, calls := 0, 0
			for _, event := range events {
				if event.Kind == "tool_start" {
					starts++
				}
				if event.Kind == "tool_call" {
					calls++
				}
			}
			if starts != 1 || calls != 0 {
				t.Fatalf("provisional stream has %d starts and %d final calls", starts, calls)
			}
		})
	}
}

func TestIncrementalFramingMatchesPrimitives(t *testing.T) {
	values := []string{
		"ordinary",
		"literal </parameter> text and </tool_call> text",
		"</parameter> \u2003<not-a-parameter> value",
		"</parameter></paraX and </parameter>\n</function> still text",
		"ends with a tag </parameter>",
	}
	var bodies []string
	for _, value := range values {
		bodies = append(bodies, " \n<function=f><parameter=x>"+value+"</parameter></function> \n</tool_call>")
	}
	bodies = append(bodies,
		"<function=f>bad </tool_call>",
		"<function=f></function><tool_call><function=g></function></tool_call>",
		"not-a-function </tool_call>",
		"<function=f><parameter=x>unfinished",
	)
	rng := rand.New(rand.NewSource(2030))
	atoms := []string{"x", " ", "\n", "\u2003", "</parameter>", "<parameter=", "</function>",
		"</tool_call>", "<tool_call>", "<", ">", "!"}
	for i := 0; i < 200; i++ {
		var body strings.Builder
		body.WriteString("<function=f><parameter=x>")
		for j := 0; j < 12; j++ {
			body.WriteString(atoms[rng.Intn(len(atoms))])
		}
		body.WriteString("</parameter></function></tool_call>")
		bodies = append(bodies, body.String())
	}
	for _, body := range bodies {
		t.Run(fmt.Sprintf("%q", body), func(t *testing.T) {
			scanner := newFrameScanner()
			var data []byte
			for _, r := range body {
				data = append(data, string(r)...)
				scanner.process(data, nil)
				call := toolparser.CallEnd(string(data))
				function := toolparser.FunctionEnd(string(data))
				if scanner.end != runeByteOffset(string(data), call) ||
					scanner.functionEnd != runeByteOffset(string(data), function) {
					t.Fatalf("prefix %q: call=(%d,%d), function=(%d,%d)",
						data, scanner.end, runeByteOffset(string(data), call),
						scanner.functionEnd, runeByteOffset(string(data), function))
				}
			}
		})
	}
}

func runeByteOffset(s string, offset int) int {
	if offset < 0 {
		return offset
	}
	for i := range s {
		if offset == 0 {
			return i
		}
		offset--
	}
	return len(s)
}
