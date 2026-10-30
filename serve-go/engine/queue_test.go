package engine

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWaitForGateProgress(t *testing.T) {
	gate := make(chan struct{}, 1)
	ticks := make(chan time.Time, 1)
	ticks <- time.Time{}
	called := false
	err := waitForGate(testContext(t), nil, gate, ticks, func() error {
		called = true
		gate <- struct{}{}
		return nil
	})
	if err != nil || !called {
		t.Fatalf("called=%t error=%v", called, err)
	}
}

func TestWaitForGateProgressFailureAndCancellation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cancel bool
	}{
		{name: "write failure"},
		{name: "client cancellation", cancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(testContext(t))
			defer cancel()
			gate := make(chan struct{}, 1)
			ticks := make(chan time.Time, 1)
			ticks <- time.Time{}
			failure := errors.New("write failed")
			err := waitForGate(ctx, nil, gate, ticks, func() error {
				if tc.cancel {
					cancel()
					return nil
				}
				return failure
			})
			want := failure
			if tc.cancel {
				want = context.Canceled
			}
			if !errors.Is(err, want) || len(gate) != 0 {
				t.Fatalf("error=%v gate=%d", err, len(gate))
			}
		})
	}
}

func TestWaitForGateShutdown(t *testing.T) {
	stop := make(chan struct{})
	close(stop)
	if err := waitForGate(testContext(t), stop, make(chan struct{}), nil, nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("error=%v", err)
	}
}
