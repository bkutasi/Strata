package engine

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSupervisorLogitBiasRestartMismatch(t *testing.T) {
	initial := startFake(t, "bias-gen", &bytes.Buffer{})
	s, err := Supervise(initial, func(ctx context.Context) (*Engine, error) {
		return Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "normal"}, io.Discard)
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close(testContext(t)) })
	if !s.SupportsLogitBias() {
		t.Fatal("initial capability missing")
	}
	initial.abort()
	select {
	case <-initial.exited:
	case <-testContext(t).Done():
		t.Fatal("initial child not reaped")
	}
	if !initial.SupportsLogitBias() || !s.SupportsLogitBias() {
		t.Fatal("lost advertised capability after exit")
	}
	_, err = s.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{Sampling: Sampling{LogitBias: map[int]float64{1: 1}}})
	if err == nil || s.SupportsLogitBias() || !s.Alive() {
		t.Fatalf("error=%v capability=%v alive=%v", err, s.SupportsLogitBias(), s.Alive())
	}
	if _, err := s.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{}); err != nil {
		t.Fatalf("replacement received rejected command: %v", err)
	}
}

func TestSupervisorQueuedCancellationDoesNotStopEngine(t *testing.T) {
	initial := startFake(t, "normal", &bytes.Buffer{})
	s, err := Supervise(initial, func(context.Context) (*Engine, error) {
		return nil, errors.New("unexpected restart")
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close(testContext(t)) })
	<-s.gate
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := s.GenerateWithOptions(ctx, []int{1}, 1, RequestOptions{OnProgress: func() error {
			cancel()
			return nil
		}})
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("queued generation error=%v", err)
		}
	case <-time.After(12 * time.Second):
		t.Fatal("queued generation did not cancel")
	}
	s.gate <- struct{}{}
	if !s.Alive() {
		t.Fatal("queued cancellation stopped the active engine")
	}
	if _, err := s.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{}); err != nil {
		t.Fatalf("engine could not serve after queued cancellation: %v", err)
	}
}

func TestSupervisorLogitBiasRestartsSupportedEngine(t *testing.T) {
	initial := startFake(t, "bias-gen", &bytes.Buffer{})
	starts := 0
	s, err := Supervise(initial, func(ctx context.Context) (*Engine, error) {
		starts++
		return Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "bias-gen"}, io.Discard)
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close(testContext(t)) })
	initial.abort()
	select {
	case <-initial.exited:
	case <-testContext(t).Done():
		t.Fatal("initial child not reaped")
	}
	if !s.SupportsLogitBias() {
		t.Fatal("lost advertised capability after exit")
	}
	_, err = s.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{
		Sampling: Sampling{LogitBias: map[int]float64{0: -100, 7: .5, 2147483647: 100}},
	})
	if err != nil || starts != 1 || !s.SupportsLogitBias() {
		t.Fatalf("error=%v starts=%d capability=%v", err, starts, s.SupportsLogitBias())
	}
}

func TestSupervisorReusesEngineAfterCancellation(t *testing.T) {
	initial, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "callback-cancel"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	var starts atomic.Int32
	s, err := Supervise(initial, func(ctx context.Context) (*Engine, error) {
		select {
		case <-initial.exited:
		default:
			return nil, errors.New("replacement started before old child was reaped")
		}
		starts.Add(1)
		return Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "normal"}, io.Discard)
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close(testContext(t)) })
	ctx, cancel := context.WithCancel(testContext(t))
	_, err = s.GenerateWithOptions(ctx, []int{1}, 1, RequestOptions{OnToken: func(int) error {
		cancel()
		return nil
	}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled generation: %v", err)
	}
	if !s.Alive() {
		t.Fatal("STOP-acknowledged cancel discarded a reusable engine")
	}
	var wg sync.WaitGroup
	failures := make(chan error, 4)
	workers := testContext(t)
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := s.GenerateWithOptions(workers, []int{12, 34}, 3, RequestOptions{})
			if err != nil || fmt.Sprint(result.Tokens) != "[42 43]" {
				failures <- fmt.Errorf("result %v: %v", result.Tokens, err)
			}
		}()
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	if starts.Load() != 0 || !s.Alive() || s.ContextLimit() != 4096 {
		t.Fatalf("starts %d, alive %t, context %d", starts.Load(), s.Alive(), s.ContextLimit())
	}
	if err := s.Close(testContext(t)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.GenerateWithOptions(testContext(t), []int{1}, 1, RequestOptions{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("generation after close: %v", err)
	}
}

func TestSupervisorCloseCancelsRestart(t *testing.T) {
	initial, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "normal"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	initial.abort()
	select {
	case <-initial.exited:
	case <-testContext(t).Done():
		t.Fatal("initial child not reaped")
	}
	started := make(chan struct{})
	s, err := Supervise(initial, func(ctx context.Context) (*Engine, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	worker := testContext(t)
	go func() {
		_, err := s.GenerateWithOptions(worker, []int{1}, 1, RequestOptions{})
		result <- err
	}()
	select {
	case <-started:
	case <-testContext(t).Done():
		t.Fatal("restart did not begin")
	}
	// The original child was deliberately killed, so its wait error is expected.
	if err := s.Close(testContext(t)); err == nil {
		t.Fatal("expected the killed child's wait error")
	}
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("restart error: %v", err)
		}
	case <-testContext(t).Done():
		t.Fatal("restart was not cancelled")
	}
	if s.Alive() {
		t.Fatal("closed supervisor reported alive")
	}
}

func TestSupervisorRejectsChangedContext(t *testing.T) {
	initial, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "normal"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	initial.abort()
	select {
	case <-initial.exited:
	case <-testContext(t).Done():
		t.Fatal("initial child not reaped")
	}
	var rejected *Engine
	starts := 0
	s, err := Supervise(initial, func(ctx context.Context) (*Engine, error) {
		starts++
		if rejected != nil {
			select {
			case <-rejected.exited:
			default:
				return nil, errors.New("rejected child not reaped")
			}
		}
		e, err := Start(ctx, os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "normal"}, io.Discard)
		if err == nil && starts == 1 {
			e.Context = 2048
			rejected = e
		}
		return e, err
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close(testContext(t)) })
	if _, err := s.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{}); err == nil {
		t.Fatal("accepted a changed context limit")
	}
	result, err := s.GenerateWithOptions(testContext(t), []int{12, 34}, 3, RequestOptions{})
	if err != nil || fmt.Sprint(result.Tokens) != "[42 43]" || starts != 2 {
		t.Fatalf("result %v, starts %d, error %v", result.Tokens, starts, err)
	}
}

func TestSupervisorDoesNotRetryUnreapedStartup(t *testing.T) {
	initial, err := Start(testContext(t), os.Args[0], []string{"-test.run=^TestHelperProcess$", "--", "normal"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	initial.abort()
	select {
	case <-initial.exited:
	case <-testContext(t).Done():
		t.Fatal("initial child not reaped")
	}
	starts := 0
	s, err := Supervise(initial, func(context.Context) (*Engine, error) {
		starts++
		return nil, ErrUnreaped
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close(testContext(t)) })
	if _, err := s.GenerateWithOptions(testContext(t), []int{1}, 1, RequestOptions{}); !errors.Is(err, ErrUnreaped) {
		t.Fatalf("restart error: %v", err)
	}
	if _, err := s.GenerateWithOptions(testContext(t), []int{1}, 1, RequestOptions{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("second request error: %v", err)
	}
	if starts != 1 || s.Alive() {
		t.Fatalf("starts %d, alive %t", starts, s.Alive())
	}
}
