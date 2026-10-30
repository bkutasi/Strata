// Package engine implements a limited, single-request native Strata pipe client.
// It is not a replacement for the Python server.
package engine

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var ErrClosed = errors.New("engine is closed")

// ErrNonfinite means a native generation window contained non-finite logits.
var ErrNonfinite = errors.New("native engine produced non-finite logits")

// EngineError is an ERR record from the native process. The process is discarded
// even for validation errors because some native ERR records are followed by DONE.
type EngineError string

func (e EngineError) Error() string { return "native engine: " + string(e) }

type record struct {
	line string
	err  error
}

// Engine owns one subprocess. Only one request or Close runs at a time.
type Engine struct {
	Context int
	CanStop bool
	Info    map[string]string

	cmd     *exec.Cmd
	stdin   io.WriteCloser
	records chan record
	exited  chan struct{}
	stop    chan struct{}
	gate    chan struct{}
	once    sync.Once
	idle    atomic.Bool
	closing atomic.Bool
	waitErr error // published by closing exited
}

// Start starts exe with the supplied arguments (which must include --serve for
// a native engine). stderr is separate from the protocol; nil discards it.
// ctx bounds startup only.
func Start(ctx context.Context, exe string, args []string, stderr io.Writer) (*Engine, error) {
	cmd := exec.Command(exe, args...)
	if stderr != nil {
		cmd.Stderr = stderr
	}
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, outWrite, err := os.Pipe()
	if err != nil {
		in.Close()
		return nil, err
	}
	cmd.Stdout = outWrite
	if err = cmd.Start(); err != nil {
		in.Close()
		out.Close()
		outWrite.Close()
		return nil, err
	}
	outWrite.Close()
	e := &Engine{cmd: cmd, stdin: in, records: make(chan record), exited: make(chan struct{}), stop: make(chan struct{}), gate: make(chan struct{}, 1), Info: map[string]string{}}
	e.gate <- struct{}{}
	go func() {
		e.waitErr = cmd.Wait()
		close(e.exited)
	}()
	go func() {
		defer out.Close()
		defer close(e.records)
		s := bufio.NewScanner(out)
		s.Buffer(make([]byte, 4096), 4*1024*1024)
		s.Split(splitRecord)
		for s.Scan() {
			line := s.Text()
			if e.idle.Load() && !e.closing.Load() {
				e.abort()
				return
			}
			fields := strings.Fields(line)
			if len(fields) > 0 && (fields[0] == "READY" || fields[0] == "DONE") {
				e.idle.Store(true)
			}
			select {
			case e.records <- record{line: line}:
			case <-e.stop:
				return
			}
		}
		readErr := s.Err()
		if readErr != nil && !e.idle.Load() {
			select {
			case e.records <- record{err: readErr}:
			case <-e.stop:
			}
		}
		if !e.closing.Load() {
			e.abort()
		}
	}()
	for {
		select {
		case <-ctx.Done():
			e.abort()
			return nil, ctx.Err()
		case r, ok := <-e.records:
			if !ok || r.err != nil {
				e.abort()
				return nil, fmt.Errorf("engine exited before READY: %w", recordError(r))
			}
			f := strings.Fields(r.line)
			if len(f) == 0 {
				continue
			}
			switch f[0] {
			case "INFO":
				for _, item := range f[1:] {
					k, v, ok := strings.Cut(item, "=")
					if ok && k != "" {
						e.Info[k] = v
					}
				}
			case "READY":
				if len(f) < 2 {
					e.abort()
					return nil, errors.New("malformed READY")
				}
				n, err := strconv.Atoi(f[1])
				if err != nil || n <= 0 {
					e.abort()
					return nil, errors.New("malformed READY context")
				}
				e.Context = n
				for _, v := range f[2:] {
					e.CanStop = e.CanStop || v == "stop"
				}
				return e, nil
			case "ERR":
				e.abort()
				return nil, EngineError(strings.TrimSpace(strings.TrimPrefix(r.line, "ERR")))
			default:
				e.abort()
				return nil, fmt.Errorf("unexpected startup record %q", r.line)
			}
		}
	}
}

// splitRecord rejects an EOF in the middle of any native protocol record.
func splitRecord(data []byte, atEOF bool) (int, []byte, error) {
	for i, b := range data {
		if b == '\n' {
			line := data[:i]
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			return i + 1, line, nil
		}
	}
	if atEOF && len(data) != 0 {
		return 0, nil, io.ErrUnexpectedEOF
	}
	return 0, nil, nil
}

func recordError(r record) error {
	if r.err != nil {
		return r.err
	}
	return io.EOF
}

func (e *Engine) abort() {
	e.once.Do(func() {
		close(e.stop)
		_ = e.cmd.Process.Kill()
		e.stdin.Close()
	})
}

// writeCommand has one writer at a time under gate. Killing the child releases a
// blocked pipe write; wait for that writer before allowing another operation.
func (e *Engine) writeCommand(ctx context.Context, command string) error {
	if err := ctx.Err(); err != nil {
		e.abort()
		return err
	}
	done := make(chan error, 1)
	go func() {
		_, err := io.WriteString(e.stdin, command)
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			e.abort()
		}
		return err
	case <-ctx.Done():
		e.abort()
		<-done
		return ctx.Err()
	}
}

func (e *Engine) acquire(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-e.gate:
		return nil
	}
}

// Result preserves the native DONE fields after the five common fields.
type Result struct {
	Tokens             []int
	Generated, Prompt  int
	PromptMS, DecodeMS float64
	Finish             string
	Extra              []string
}

// Generate sends a greedy text-only GEN. Cancellation sends STOP if advertised,
// then waits at most one second for DONE before killing the process. A cancelled
// request always discards the process: STOP can race native request admission.
func (e *Engine) Generate(ctx context.Context, ids []int, maxNew int) (Result, error) {
	var result Result
	if maxNew < 1 || len(ids) == 0 {
		return result, errors.New("GEN requires positive maxNew and at least one token")
	}
	tokens := make([]string, len(ids))
	for i, id := range ids {
		if id < 0 || int64(id) > 2147483647 {
			return result, errors.New("token ID outside nonnegative int32 range")
		}
		tokens[i] = strconv.Itoa(id)
	}
	if err := e.acquire(ctx); err != nil {
		return result, err
	}
	defer func() { e.gate <- struct{}{} }()
	select {
	case <-e.exited:
		return result, ErrClosed
	default:
	}
	e.idle.Store(false)
	if err := e.writeCommand(ctx, fmt.Sprintf("GEN %d %s\n", maxNew, strings.Join(tokens, ","))); err != nil {
		return result, err
	}
	var cancel <-chan time.Time
	var timer *time.Timer
	ctxDone := ctx.Done()
	var cancelled error
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()
	for {
		select {
		case <-ctxDone:
			cancelled = ctx.Err()
			ctxDone = nil
			if !e.CanStop {
				e.abort()
				return result, cancelled
			}
			stopCtx, stopCancel := context.WithTimeout(context.Background(), time.Second)
			err := e.writeCommand(stopCtx, "STOP\n")
			stopCancel()
			if err != nil {
				return result, errors.Join(cancelled, err)
			}
			timer = time.NewTimer(time.Second)
			cancel = timer.C
		case <-cancel:
			e.abort()
			return result, cancelled
		case r, ok := <-e.records:
			if !ok || r.err != nil {
				e.abort()
				if cancelled != nil {
					return result, errors.Join(cancelled, recordError(r))
				}
				return result, fmt.Errorf("engine output ended: %w", recordError(r))
			}
			f := strings.Fields(r.line)
			if len(f) == 0 {
				e.abort()
				return result, errors.New("empty engine record")
			}
			switch f[0] {
			case "T":
				if len(f) != 2 {
					e.abort()
					return result, fmt.Errorf("malformed token record %q", r.line)
				}
				id, err := strconv.ParseInt(f[1], 10, 32)
				if err != nil || id < 0 {
					e.abort()
					return result, fmt.Errorf("malformed token record %q", r.line)
				}
				result.Tokens = append(result.Tokens, int(id))
			case "PP", "RESUME", "REUSED":
				// Progress and cache reuse are intentionally not exposed.
			case "ERR":
				e.abort()
				return result, EngineError(strings.TrimSpace(strings.TrimPrefix(r.line, "ERR")))
			case "DONE":
				if err := parseDone(f, &result); err != nil {
					e.abort()
					return result, err
				}
				if cancelled != nil {
					e.abort()
					return result, cancelled
				}
				return result, nil
			default:
				e.abort()
				return result, fmt.Errorf("unsupported engine record %q", r.line)
			}
		}
	}
}

func parseDone(f []string, r *Result) error {
	if len(f) < 6 {
		return errors.New("malformed DONE")
	}
	var err error
	if r.Generated, err = strconv.Atoi(f[1]); err != nil || r.Generated < 0 {
		return errors.New("malformed DONE generated count")
	}
	if r.Prompt, err = strconv.Atoi(f[2]); err != nil || r.Prompt < 0 {
		return errors.New("malformed DONE prompt count")
	}
	if r.PromptMS, err = strconv.ParseFloat(f[3], 64); err != nil || r.PromptMS < 0 || math.IsNaN(r.PromptMS) || math.IsInf(r.PromptMS, 0) {
		return errors.New("malformed DONE prompt time")
	}
	if r.DecodeMS, err = strconv.ParseFloat(f[4], 64); err != nil || r.DecodeMS < 0 || math.IsNaN(r.DecodeMS) || math.IsInf(r.DecodeMS, 0) {
		return errors.New("malformed DONE decode time")
	}
	if f[5] == "nonfinite" {
		r.Finish = f[5]
		r.Extra = append([]string(nil), f[6:]...)
		return ErrNonfinite
	}
	if f[5] != "stop" && f[5] != "length" && f[5] != "cancel" {
		return errors.New("unsupported DONE finish")
	}
	r.Finish = f[5]
	r.Extra = append([]string(nil), f[6:]...)
	return nil
}

// Close sends QUIT and waits for the child. If ctx expires, it kills the child.
func (e *Engine) Close(ctx context.Context) error {
	if err := e.acquire(ctx); err != nil {
		e.abort()
		return err
	}
	defer func() { e.gate <- struct{}{} }()
	select {
	case <-e.exited:
		return e.waitErr
	default:
	}
	e.closing.Store(true)
	if err := e.writeCommand(ctx, "QUIT\n"); err != nil {
		return err
	}
	// Drain unexpected output so the reader can reach EOF and reap the child.
	for {
		select {
		case <-ctx.Done():
			e.abort()
			return ctx.Err()
		case _, ok := <-e.records:
			if !ok {
				select {
				case <-e.exited:
					return e.waitErr
				case <-ctx.Done():
					e.abort()
					return ctx.Err()
				}
			}
		}
	}
}
