package engine

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
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
	fmt.Println("READY 4096 stop")
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
			switch mode {
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
