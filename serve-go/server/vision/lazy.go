package vision

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

// Lazy starts a vision reader for each image instead of keeping one resident.
// This saves memory between images at the cost of startup time for each image.
type Lazy struct {
	exe    string
	args   []string
	env    []string
	lock   chan struct{}
	mu     sync.Mutex
	closed bool
	cancel context.CancelFunc
}

// NewLazy creates a reader without starting a process. A nil env is inherited.
func NewLazy(exe string, args, env []string) *Lazy {
	var savedEnv []string
	if env != nil {
		savedEnv = append([]string{}, env...)
	}
	l := &Lazy{exe: exe, args: append([]string(nil), args...), env: savedEnv, lock: make(chan struct{}, 1)}
	l.lock <- struct{}{}
	return l
}

// Encode returns an independently owned SVE file. The caller must remove it.
func (l *Lazy) Encode(ctx context.Context, image []byte) (path string, tokens int, err error) {
	select {
	case <-ctx.Done():
		return "", 0, ctx.Err()
	case <-l.lock:
	}
	defer func() { l.lock <- struct{}{} }()

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return "", 0, fmt.Errorf("vision reader is closed")
	}
	ctx, cancel := context.WithCancel(ctx)
	l.cancel = cancel
	l.mu.Unlock()
	defer func() {
		l.mu.Lock()
		l.cancel = nil
		l.mu.Unlock()
		cancel()
	}()

	r, err := StartWithEnv(ctx, l.exe, l.args, l.env)
	if err != nil {
		return "", 0, err
	}
	defer func() {
		err = errorsJoin(err, r.Close())
		if err != nil && path != "" {
			_ = os.Remove(path)
			path, tokens = "", 0
		}
	}()
	source, tokens, err := r.Encode(ctx, image)
	if err != nil {
		return "", 0, err
	}
	in, err := os.Open(source)
	if err != nil {
		return "", 0, err
	}
	defer in.Close()
	out, err := os.CreateTemp("", "strata-vision-*.sve")
	if err != nil {
		return "", 0, err
	}
	path = out.Name()
	if strings.ContainsAny(path, " \t\r\n=") {
		err = fmt.Errorf("vision temporary path contains characters unsafe for GENI: %q", path)
	} else if _, err = io.Copy(out, in); err == nil {
		err = ctx.Err()
	}
	err = errorsJoin(err, out.Close())
	if err != nil {
		return path, 0, err
	}
	return path, tokens, nil
}

// Close cancels any active encode and waits until its child has been reaped.
func (l *Lazy) Close() error {
	l.mu.Lock()
	l.closed = true
	if l.cancel != nil {
		l.cancel()
	}
	l.mu.Unlock()
	<-l.lock
	l.lock <- struct{}{}
	return nil
}
