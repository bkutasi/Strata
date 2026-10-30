package engine

import (
	"context"
	"time"
)

func acquireWithProgress(ctx context.Context, stop <-chan struct{}, gate <-chan struct{}, progress func() error) error {
	var ticks <-chan time.Time
	if progress != nil {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		ticks = ticker.C
	}
	return waitForGate(ctx, stop, gate, ticks, progress)
}

func waitForGate(ctx context.Context, stop <-chan struct{}, gate <-chan struct{}, ticks <-chan time.Time, progress func() error) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-stop:
			return ErrClosed
		case <-gate:
			return nil
		case <-ticks:
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := progress(); err != nil {
				return err
			}
		}
	}
}
