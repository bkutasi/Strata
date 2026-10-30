package engine

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// The test binary is the child: all synchronization is via pipes, not sleeps.
func TestHelperProcess(t *testing.T) {
	if len(os.Args) < 4 || os.Args[2] != "--" {
		return
	}
	mode := os.Args[3]
	if mode == "startup-eof" {
		os.Exit(0)
	}
	if mode == "truncated-ready" {
		fmt.Print("READY 4096 stop")
		os.Exit(0)
	}
	if mode == "startup-closed-stdout" {
		os.Stdout.Close()
		_, _ = io.Copy(io.Discard, os.Stdin)
		os.Exit(0)
	}
	fmt.Fprintln(os.Stderr, "child diagnostics")
	fmt.Println("INFO engine=test context=4096")
	if mode == "early-no-stop" {
		fmt.Println("READY 4096")
	} else {
		fmt.Println("READY 4096 stop")
	}
	if mode == "blocked-write" {
		var b [1]byte
		if _, err := os.Stdin.Read(b[:]); err != nil {
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "request accepted")
		r, w, err := os.Pipe()
		if err != nil {
			os.Exit(2)
		}
		defer w.Close()
		_, _ = r.Read(b[:])
		os.Exit(2)
	}
	s := bufio.NewScanner(os.Stdin)
	requests := 0
	for s.Scan() {
		line := s.Text()
		if line == "TRIGGER" && mode == "idle-eof" {
			os.Stdout.Close()
			_, _ = io.Copy(io.Discard, os.Stdin)
			os.Exit(0)
		}
		if line == "TRIGGER" && mode == "idle-output" {
			fmt.Println("T 123")
			os.Exit(0)
		}
		if line == "QUIT" {
			os.Exit(0)
		}
		if strings.HasPrefix(line, "GEN ") {
			requests++
			switch mode {
			case "early-stop":
				if requests == 1 {
					fmt.Println("T 42")
					if !s.Scan() || s.Text() != "STOP" {
						os.Exit(2)
					}
					fmt.Println("T 43")
					fmt.Println("DONE 2 1 1 2 cancel")
				} else {
					fmt.Println("T 44")
					fmt.Println("DONE 1 1 1 2 stop")
				}
			case "early-no-stop":
				fmt.Println("T 42")
				fmt.Println("T 43")
				fmt.Println("DONE 2 1 1 2 length")
			case "sampling":
				if line != "GEN 3 temperature=0.7 top_p=0.9 top_k=64 min_p=0.1 penalty_repeat=1.1 penalty_freq=-0.5 penalty_present=0.2 penalty_last_n=64 seed=123 12,34" {
					fmt.Println("ERR unexpected command")
					continue
				}
				fmt.Println("T 42")
				if !s.Scan() || s.Text() != "ACK" {
					os.Exit(2)
				}
				fmt.Println("T 43")
				fmt.Println("DONE 2 2 1 2 stop")
			case "callback-error", "callback-cancel", "callback-panic":
				fmt.Println("T 42")
				if mode == "callback-cancel" {
					if !s.Scan() || s.Text() != "STOP" {
						os.Exit(2)
					}
					fmt.Println("T 43")
					fmt.Println("DONE 2 2 1 2 cancel")
				} else {
					_, _ = io.Copy(io.Discard, os.Stdin)
				}
			case "done-only":
				fmt.Println("DONE 0 1 0 1 stop")
			case "error":
				fmt.Println("ERR verify: timed out at layer 3")
				fmt.Println("DONE 0 1 0 1 error")
			case "eof":
				os.Exit(0)
			case "closed-stdout":
				os.Stdout.Close()
				_, _ = io.Copy(io.Discard, os.Stdin)
				os.Exit(0)
			case "nonfinite":
				fmt.Println("T 42")
				fmt.Println("DONE 1 1 1 2 nonfinite 0 0 0")
			case "truncated-done":
				fmt.Print("DONE 0 1 0 1 stop")
				os.Exit(0)
			case "nan-prompt":
				fmt.Println("DONE 0 1 NaN 1 stop")
			case "inf-prompt":
				fmt.Println("DONE 0 1 +Inf 1 stop")
			case "nan-decode":
				fmt.Println("DONE 0 1 1 NaN stop")
			case "inf-decode":
				fmt.Println("DONE 0 1 1 +Inf stop")
			case "bad":
				fmt.Println("T not-an-id")
			case "cancel":
				fmt.Fprintln(os.Stderr, "request accepted")
			default:
				if line != "GEN 3 12,34" {
					fmt.Println("ERR unexpected command")
					continue
				}
				fmt.Println("RESUME 1")
				fmt.Println("PP 1 2 1 2")
				fmt.Println("REUSED 1")
				fmt.Println("T 42")
				fmt.Println("T 43")
				fmt.Println("DONE 2 2 1.5 2.5 stop 0 0 1 chunk=2")
			}
		} else if line == "STOP" && mode == "cancel" {
			fmt.Println("DONE 0 2 0 1 cancel 0 0 0 0 0")
		}
	}
	os.Exit(0)
}

func TestStopGenerationDrainsAndReusesEngine(t *testing.T) {
	for _, mode := range []string{"early-stop", "early-no-stop"} {
		t.Run(mode, func(t *testing.T) {
			var stderr bytes.Buffer
			e := startFake(t, mode, &stderr)
			var seen []int
			result, err := e.GenerateWithOptions(testContext(t), []int{1}, 3, RequestOptions{
				OnToken: func(id int) error {
					seen = append(seen, id)
					return ErrStopGeneration
				},
			})
			if err != nil || result.Finish != "stop" || fmt.Sprint(seen) != "[42]" || !e.Alive() {
				t.Fatalf("result=%+v seen=%v alive=%v error=%v", result, seen, e.Alive(), err)
			}
			seen = nil
			result, err = e.GenerateWithOptions(testContext(t), []int{1}, 3, RequestOptions{
				OnToken: func(id int) error { seen = append(seen, id); return nil },
			})
			want := "[44]"
			if mode == "early-no-stop" {
				want = "[42 43]"
			}
			if err != nil || fmt.Sprint(seen) != want || !e.Alive() {
				t.Fatalf("second result=%+v seen=%v alive=%v error=%v", result, seen, e.Alive(), err)
			}
		})
	}
}

func TestCancellationAfterStopDiscardsEngine(t *testing.T) {
	var stderr bytes.Buffer
	e := startFake(t, "early-stop", &stderr)
	ctx, cancel := context.WithCancel(testContext(t))
	defer cancel()
	_, err := e.GenerateWithOptions(ctx, []int{1}, 3, RequestOptions{
		OnToken: func(int) error {
			cancel()
			return ErrStopGeneration
		},
	})
	if !errors.Is(err, context.Canceled) || e.Alive() {
		t.Fatalf("error=%v alive=%v", err, e.Alive())
	}
}

func TestCallbackPanicDiscardsChild(t *testing.T) {
	var stderr bytes.Buffer
	e := startFake(t, "callback-panic", &stderr)
	marker := errors.New("callback panic")
	func() {
		defer func() {
			if got := recover(); got != marker {
				t.Fatalf("panic = %v, want %v", got, marker)
			}
		}()
		_, _ = e.GenerateWithOptions(testContext(t), []int{1}, 1, RequestOptions{
			OnToken: func(int) error { panic(marker) },
		})
		t.Fatal("callback panic did not propagate")
	}()
	select {
	case <-e.exited:
		if e.cmd.ProcessState == nil {
			t.Fatal("child was not reaped")
		}
	case <-testContext(t).Done():
		t.Fatal("panicking callback left child running")
	}
	if _, err := e.Generate(testContext(t), []int{1}, 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("panicked engine reused: %v", err)
	}
}

// cancelOnFinalCheck makes cancellation coincide with the DONE completion
// check, after the pre-select check has already returned nil.
type cancelOnFinalCheck struct {
	context.Context
	cancel context.CancelFunc
	checks int
}

func (c *cancelOnFinalCheck) Err() error {
	c.checks++
	if c.checks == 3 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCancellationAtDoneDiscardsChild(t *testing.T) {
	var stderr bytes.Buffer
	e := startFake(t, "done-only", &stderr)
	base, cancel := context.WithCancel(testContext(t))
	defer cancel()
	ctx := &cancelOnFinalCheck{Context: base, cancel: cancel}
	r, err := e.Generate(ctx, []int{1}, 1)
	if !errors.Is(err, context.Canceled) || r.Finish != "stop" {
		t.Fatalf("result %#v, error %v", r, err)
	}
	select {
	case <-e.exited:
	case <-testContext(t).Done():
		t.Fatal("cancelled child not reaped")
	}
	if _, err := e.Generate(testContext(t), []int{1}, 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("cancelled engine reused: %v", err)
	}
}

func TestSamplingAndOrderedCallbacks(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	seed := uint64(123)
	var stderr bytes.Buffer
	e := startFake(t, "sampling", &stderr)
	var seen []int
	r, err := e.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{
		Sampling: Sampling{Temperature: f(.7), TopP: f(.9), TopK: i(0), MinP: f(.1), RepetitionPenalty: f(1.1), FrequencyPenalty: f(-.5), PresencePenalty: f(.2), Seed: &seed},
		OnToken: func(id int) error {
			seen = append(seen, id)
			if id == 42 {
				_, err := io.WriteString(e.stdin, "ACK\n")
				return err
			}
			return nil
		},
	})
	if err != nil || fmt.Sprint(seen) != "[42 43]" || fmt.Sprint(r.Tokens) != "[42 43]" || r.Finish != "stop" {
		t.Fatalf("result %#v, callbacks %v, error %v", r, seen, err)
	}
}

func TestSamplingDefaultsAndValidation(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	zero := uint64(0)
	keys, err := samplingKeys(Sampling{Temperature: f(0), TopP: f(1), MinP: f(0), RepetitionPenalty: f(1), FrequencyPenalty: f(0), PresencePenalty: f(0), PenaltyLastN: i(9), Seed: &zero})
	if err != nil || keys != "" {
		t.Fatalf("defaults: %q %v", keys, err)
	}
	keys, err = samplingKeys(Sampling{TopK: i(100), FrequencyPenalty: f(1), PenaltyLastN: i(12)})
	if err != nil || keys != " top_k=64 penalty_freq=1 penalty_last_n=12" {
		t.Fatalf("mapping: %q %v", keys, err)
	}
	for _, s := range []Sampling{{Temperature: f(math.NaN())}, {TopP: f(0)}, {MinP: f(1.1)}, {TopK: i(-1)}, {PenaltyLastN: i(-1)}, {RepetitionPenalty: f(0)}, {FrequencyPenalty: f(math.Inf(1))}} {
		if _, err := samplingKeys(s); err == nil {
			t.Fatalf("accepted invalid sampling: %#v", s)
		}
	}
	var stderr bytes.Buffer
	e := startFake(t, "normal", &stderr)
	if _, err := e.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{Sampling: Sampling{TopP: f(0)}}); err == nil {
		t.Fatal("accepted invalid request")
	}
	if _, err := e.Generate(testContext(t), []int{12, 34}, 3); err != nil {
		t.Fatalf("validation damaged engine: %v", err)
	}
}

func TestCallbackFailureAndCancellation(t *testing.T) {
	for _, mode := range []string{"callback-error", "callback-cancel"} {
		t.Run(mode, func(t *testing.T) {
			var stderr bytes.Buffer
			e := startFake(t, mode, &stderr)
			ctx, cancel := context.WithCancel(testContext(t))
			defer cancel()
			failure := errors.New("callback failure")
			var seen []int
			r, err := e.GenerateWithOptions(ctx, []int{1}, 3, RequestOptions{OnToken: func(id int) error {
				seen = append(seen, id)
				if mode == "callback-cancel" {
					cancel()
					return nil
				}
				return failure
			}})
			want := failure
			if mode == "callback-cancel" {
				want = context.Canceled
			}
			if !errors.Is(err, want) || fmt.Sprint(seen) != "[42]" || len(r.Tokens) == 0 || r.Tokens[0] != 42 {
				t.Fatalf("result %#v callbacks %v error %v", r, seen, err)
			}
			if mode == "callback-cancel" && (fmt.Sprint(r.Tokens) != "[42 43]" || r.Finish != "cancel") {
				t.Fatalf("drained result %#v", r)
			}
			select {
			case <-e.exited:
			case <-testContext(t).Done():
				t.Fatal("child not reaped")
			}
		})
	}
}

func TestBlockedGENWriteCancellationReapsChild(t *testing.T) {
	w := signalWriter{accepted: make(chan struct{})}
	e, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "blocked-write"}, w)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = e.Close(testContext(t)) })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	ids := make([]int, 1<<20)
	go func() {
		_, err := e.Generate(ctx, ids, 1)
		done <- err
	}()
	select {
	case <-w.accepted:
	case <-testContext(t).Done():
		t.Fatal("child did not observe GEN")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("blocked write: %v", err)
		}
	case <-testContext(t).Done():
		t.Fatal("blocked GEN write did not return")
	}
	select {
	case <-e.exited:
		if e.cmd.ProcessState == nil || e.cmd.ProcessState.Success() {
			t.Fatal("blocked child was not killed and reaped")
		}
	case <-testContext(t).Done():
		t.Fatal("blocked child was not reaped")
	}
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func startFake(t *testing.T, mode string, stderr *bytes.Buffer) *Engine {
	t.Helper()
	e, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", mode}, stderr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = e.Close(testContext(t)) })
	return e
}

func TestNativeRecordsAndQuit(t *testing.T) {
	var stderr bytes.Buffer
	e := startFake(t, "normal", &stderr)
	if e.Context != 4096 || !e.CanStop || e.Info["engine"] != "test" {
		t.Fatalf("startup: %#v", e)
	}
	for i := 0; i < 2; i++ {
		r, err := e.Generate(testContext(t), []int{12, 34}, 3)
		if err != nil || fmt.Sprint(r.Tokens) != "[42 43]" || r.Generated != 2 || r.Prompt != 2 || r.PromptMS != 1.5 || r.DecodeMS != 2.5 || r.Finish != "stop" || len(r.Extra) != 4 {
			t.Fatalf("result %#v, error %v", r, err)
		}
	}
	if err := e.Close(testContext(t)); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stderr.String(), "child diagnostics") {
		t.Fatal("stderr was not kept separate")
	}
	if _, err := e.Generate(testContext(t), []int{1}, 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("generate after close: %v", err)
	}
}

func TestFailuresInvalidateProcess(t *testing.T) {
	for _, mode := range []string{"error", "eof", "bad", "closed-stdout", "nonfinite", "truncated-done", "nan-prompt", "inf-prompt", "nan-decode", "inf-decode"} {
		t.Run(mode, func(t *testing.T) {
			var stderr bytes.Buffer
			e := startFake(t, mode, &stderr)
			result, err := e.Generate(testContext(t), []int{1}, 1)
			if err == nil {
				t.Fatal("expected failure")
			}
			if mode == "error" {
				var native EngineError
				if !errors.As(err, &native) || !strings.Contains(string(native), "verify") {
					t.Fatalf("expected native error, got %v", err)
				}
			}
			if mode == "closed-stdout" && !errors.Is(err, io.EOF) {
				t.Fatalf("expected stdout EOF, got %v", err)
			}
			if mode == "truncated-done" && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("expected truncated record, got %v", err)
			}
			if mode == "nonfinite" && (!errors.Is(err, ErrNonfinite) || result.Finish != "nonfinite" || fmt.Sprint(result.Tokens) != "[42]") {
				t.Fatalf("nonfinite result %#v, error %v", result, err)
			}
			select {
			case <-e.exited:
			case <-testContext(t).Done():
				t.Fatal("child was not reaped")
			}
			if _, err := e.Generate(testContext(t), []int{1}, 1); !errors.Is(err, ErrClosed) {
				t.Fatalf("process reused after failure: %v", err)
			}
		})
	}
}

type signalWriter struct{ accepted chan struct{} }

func (w signalWriter) Write(p []byte) (int, error) {
	if strings.Contains(string(p), "request accepted") {
		close(w.accepted)
	}
	return len(p), nil
}

func TestCancellationSendsStop(t *testing.T) {
	w := signalWriter{accepted: make(chan struct{})}
	e, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "cancel"}, w)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	type response struct {
		result Result
		err    error
	}
	done := make(chan response, 1)
	go func() {
		result, err := e.Generate(ctx, []int{1, 2}, 3)
		done <- response{result, err}
	}()
	select {
	case <-w.accepted:
	case <-testContext(t).Done():
		t.Fatal("child did not accept request")
	}
	cancel()
	select {
	case got := <-done:
		if !errors.Is(got.err, context.Canceled) || got.result.Finish != "cancel" {
			t.Fatalf("cancellation: %#v, %v", got.result, got.err)
		}
	case <-testContext(t).Done():
		t.Fatal("cancellation did not complete")
	}
	select {
	case <-e.exited:
	case <-testContext(t).Done():
		t.Fatal("cancelled child not reaped")
	}
}

func TestStartupEOFAndValidation(t *testing.T) {
	for _, mode := range []string{"startup-eof", "startup-closed-stdout"} {
		_, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", mode}, nil)
		if !errors.Is(err, io.EOF) {
			t.Fatalf("%s: expected startup EOF, got %v", mode, err)
		}
	}
	_, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "truncated-ready"}, nil)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("truncated READY: %v", err)
	}
	var stderr bytes.Buffer
	e := startFake(t, "normal", &stderr)
	if _, err := e.Generate(testContext(t), []int{-1}, 1); err == nil {
		t.Fatal("negative token accepted")
	}
}

func TestIdleOutputAndEOFReapWithoutAnotherCall(t *testing.T) {
	for _, mode := range []string{"idle-eof", "idle-output"} {
		t.Run(mode, func(t *testing.T) {
			var stderr bytes.Buffer
			e := startFake(t, mode, &stderr)
			if _, err := io.WriteString(e.stdin, "TRIGGER\n"); err != nil {
				t.Fatal(err)
			}
			select {
			case <-e.exited:
				if e.cmd.ProcessState == nil {
					t.Fatal("child was not reaped")
				}
				if mode == "idle-eof" && e.cmd.ProcessState.Success() {
					t.Fatal("child with closed stdout was not killed")
				}
			case <-testContext(t).Done():
				t.Fatal("idle child was not reaped")
			}
		})
	}
}
