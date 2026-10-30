package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

func TestSupervisorRestartsOnceAfterCancellation(t *testing.T) {
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
	if s.Alive() {
		t.Fatal("discarded engine reported alive")
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
	if starts.Load() != 1 || !s.Alive() || s.ContextLimit() != 4096 {
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
