package engine

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
)

// Supervisor restarts a discarded engine before the next request. It never
// retries a generation that has already started.
type Supervisor struct {
	current atomic.Pointer[Engine]
	closed  atomic.Bool
	gate    chan struct{}
	stop    context.Context
	cancel  context.CancelFunc
	start   func(context.Context) (*Engine, error)
	limit   int
}

// Supervise takes ownership of initial and uses start to replace it if it exits.
// start must honor its context and return ErrUnreaped if it cannot confirm that
// a child from a failed start has exited.
func Supervise(initial *Engine, start func(context.Context) (*Engine, error)) (*Supervisor, error) {
	if initial == nil || start == nil {
		return nil, errors.New("engine supervisor requires an engine and start function")
	}
	stop, cancel := context.WithCancel(context.Background())
	s := &Supervisor{gate: make(chan struct{}, 1), stop: stop, cancel: cancel, start: start, limit: initial.Context}
	s.gate <- struct{}{}
	s.current.Store(initial)
	return s, nil
}

func (s *Supervisor) ContextLimit() int { return s.limit }

func (s *Supervisor) Alive() bool {
	return !s.closed.Load() && s.current.Load().Alive()
}

func (s *Supervisor) GenerateWithOptions(ctx context.Context, ids []int, maxNew int, opts RequestOptions) (Result, error) {
	var result Result
	if err := ctx.Err(); err != nil {
		return result, err
	}
	select {
	case <-ctx.Done():
		return result, ctx.Err()
	case <-s.stop.Done():
		return result, ErrClosed
	case <-s.gate:
	}
	defer func() { s.gate <- struct{}{} }()
	if s.closed.Load() {
		return result, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	e := s.current.Load()
	if !e.Alive() {
		// The old process must be reaped before loading another model.
		select {
		case <-e.exited:
		case <-ctx.Done():
			return result, ctx.Err()
		case <-s.stop.Done():
			return result, ErrClosed
		}
		startCtx, cancel := context.WithCancel(ctx)
		stopCancel := context.AfterFunc(s.stop, cancel)
		fresh, err := s.start(startCtx)
		stopCancel()
		cancel()
		if err != nil {
			if errors.Is(err, ErrUnreaped) {
				s.closed.Store(true)
				s.cancel()
			}
			return result, fmt.Errorf("restart engine: %w", err)
		}
		if fresh == nil {
			return result, errors.New("restart engine returned no process")
		}
		s.current.Store(fresh)
		if s.closed.Load() {
			fresh.abort()
			return result, ErrClosed
		}
		if err := ctx.Err(); err != nil {
			fresh.abort()
			return result, err
		}
		if fresh.Context != s.limit {
			fresh.abort()
			return result, errors.New("restarted engine context changed")
		}
		e = fresh
	}
	return e.GenerateWithOptions(ctx, ids, maxNew, opts)
}

// Close prevents restarts and closes the currently owned process.
func (s *Supervisor) Close(ctx context.Context) error {
	s.closed.Store(true)
	s.cancel()
	select {
	case <-s.gate:
		defer func() { s.gate <- struct{}{} }()
		return s.current.Load().Close(ctx)
	case <-ctx.Done():
		s.current.Load().abort()
		return ctx.Err()
	}
}
