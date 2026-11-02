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
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var ErrClosed = errors.New("engine is closed")

// ErrUnreaped means a failed startup's child did not exit after it was killed.
var ErrUnreaped = errors.New("failed engine startup was not reaped")

// ErrNonfinite means a native generation window contained non-finite logits.
var ErrNonfinite = errors.New("native engine produced non-finite logits")

// ErrStopGeneration asks the engine to stop after the current token and drain
// its DONE record, so the process can serve the next request.
var ErrStopGeneration = errors.New("stop generation")

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

// Alive reports whether the child is still available for requests.
func (e *Engine) Alive() bool {
	if e.closing.Load() {
		return false
	}
	select {
	case <-e.exited:
		return false
	case <-e.stop:
		return false
	default:
		return true
	}
}

func (e *Engine) ContextLimit() int { return e.Context }

// SupportsLogitBias reports the last advertised capability, even after exit.
// Batching does not apply per-request bias.
func (e *Engine) SupportsLogitBias() bool {
	if e.Info["logit_bias"] != "1" {
		return false
	}
	if slots, ok := e.Info["batch_slots"]; ok {
		n, err := strconv.Atoi(slots)
		return err == nil && n == 0
	}
	return true
}

// Start starts exe with the supplied arguments (which must include --serve for
// a native engine). stderr is separate from the protocol; nil discards it.
// ctx bounds startup only.
func Start(ctx context.Context, exe string, args []string, stderr io.Writer) (_ *Engine, err error) {
	return StartWithProcessConfig(ctx, exe, args, stderr, ProcessConfig{})
}

// ProcessConfig sets the engine child's working directory and environment.
// A nil Env inherits the current process environment.
type ProcessConfig struct {
	Dir string
	Env []string
}

// StartWithProcessConfig starts the engine using the supplied child settings.
func StartWithProcessConfig(ctx context.Context, exe string, args []string, stderr io.Writer, config ProcessConfig) (_ *Engine, err error) {
	cmd := exec.Command(exe, args...)
	cmd.Dir = config.Dir
	cmd.Env = config.Env
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
	defer func() {
		if err == nil {
			return
		}
		e.abort()
		timer := time.NewTimer(3 * time.Second)
		defer timer.Stop()
		select {
		case <-e.exited:
		case <-timer.C:
			err = errors.Join(err, ErrUnreaped)
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

// Sampling uses nil fields for native defaults. Values are validated before
// sending; TopK 0 or greater than 64 maps to 64 as in the Python server.
type Sampling struct {
	Temperature, TopP, MinP                              *float64
	TopK, PenaltyLastN                                   *int
	RepetitionPenalty, FrequencyPenalty, PresencePenalty *float64
	Seed                                                 *uint64
	LogitBias                                            map[int]float64
	Logprobs                                             *int
	AllowedTokenIDs                                      []int
	DisableCheckpoint                                    bool
	SpeedProjection                                      *bool
	TunePCIEFrac, TuneSpecMinP, TunePrefillPipe          *float64
	TuneAuxCPUs                                          *bool
}

// Logprob describes the sampled token and the ordered native alternatives.
type Logprob struct {
	Value        float64
	Alternatives []LogprobAlternative
}

type LogprobAlternative struct {
	TokenID int
	Value   float64
}

// RequestOptions configures a text request. OnProgress runs synchronously while
// waiting for the engine, for native PP records and periodically during generation.
// OnPromptProgress receives the native prompt position and total, when available.
// OnToken runs synchronously in protocol order, after the token is appended to
// Result and before DONE.
// OnLogprob runs immediately before OnToken, with nil if no LP was emitted.
// Callbacks must not call Generate or Close on the same Engine.
type RequestOptions struct {
	Sampling         Sampling
	EmbeddingsPath   string
	OnProgress       func() error
	OnPromptProgress func(int, int)
	OnToken          func(int) error
	OnLogprob        func(int, *Logprob) error
}

func samplingKeys(s Sampling) (string, error) {
	var keys strings.Builder
	addFloat := func(key string, v *float64, min, max float64, include func(float64) bool) error {
		if v == nil {
			return nil
		}
		if math.IsNaN(*v) || math.IsInf(*v, 0) || *v < min || *v > max {
			return fmt.Errorf("invalid %s", key)
		}
		if include(*v) {
			keys.WriteString(" " + key + "=" + strconv.FormatFloat(*v, 'g', -1, 64))
		}
		return nil
	}
	positive := func(v float64) bool { return v > 0 }
	if err := addFloat("temperature", s.Temperature, 0, math.MaxFloat32, positive); err != nil {
		return "", err
	}
	if err := addFloat("top_p", s.TopP, math.SmallestNonzeroFloat32, 1, func(v float64) bool { return v < 1 }); err != nil {
		return "", err
	}
	if s.TopK != nil {
		if *s.TopK < 0 || int64(*s.TopK) > math.MaxInt32 {
			return "", errors.New("invalid top_k")
		}
		k := *s.TopK
		if k == 0 || k > 64 {
			k = 64
		}
		fmt.Fprintf(&keys, " top_k=%d", k)
	}
	if err := addFloat("min_p", s.MinP, 0, 1, positive); err != nil {
		return "", err
	}
	for _, tune := range []struct {
		key   string
		value *float64
		max   float64
	}{
		{"pcie_frac", s.TunePCIEFrac, 1},
		{"spec_min_p", s.TuneSpecMinP, 1},
		{"pipe", s.TunePrefillPipe, 262144},
	} {
		if err := addFloat(tune.key, tune.value, 0, tune.max, func(float64) bool { return true }); err != nil {
			return "", err
		}
	}
	penalty := false
	for _, p := range []struct {
		name  string
		value *float64
		min   float64
		def   float64
	}{
		{"penalty_repeat", s.RepetitionPenalty, math.SmallestNonzeroFloat32, 1},
		{"penalty_freq", s.FrequencyPenalty, -math.MaxFloat32, 0},
		{"penalty_present", s.PresencePenalty, -math.MaxFloat32, 0},
	} {
		if err := addFloat(p.name, p.value, p.min, math.MaxFloat32, func(v float64) bool { return v != p.def }); err != nil {
			return "", err
		}
		penalty = penalty || (p.value != nil && *p.value != p.def)
	}
	if s.PenaltyLastN != nil && (*s.PenaltyLastN < 0 || int64(*s.PenaltyLastN) > math.MaxInt32) {
		return "", errors.New("invalid penalty_last_n")
	}
	if penalty {
		n := 64
		if s.PenaltyLastN != nil && *s.PenaltyLastN > 0 {
			n = *s.PenaltyLastN
		}
		fmt.Fprintf(&keys, " penalty_last_n=%d", n)
	}
	if s.Seed != nil && *s.Seed > 0 {
		fmt.Fprintf(&keys, " seed=%d", *s.Seed)
	}
	if s.DisableCheckpoint {
		keys.WriteString(" ckpt=0")
	}
	if s.SpeedProjection != nil {
		if *s.SpeedProjection {
			keys.WriteString(" cvec=1")
		} else {
			keys.WriteString(" cvec=0")
		}
	}
	if s.TuneAuxCPUs != nil {
		if *s.TuneAuxCPUs {
			keys.WriteString(" aux_cpus=1")
		} else {
			keys.WriteString(" aux_cpus=0")
		}
	}
	if s.Logprobs != nil {
		if *s.Logprobs < 0 || *s.Logprobs > 20 {
			return "", errors.New("invalid logprobs")
		}
		fmt.Fprintf(&keys, " logprobs=%d", *s.Logprobs)
		if len(s.AllowedTokenIDs) > 0 {
			keys.WriteString(" allowed=")
			for i, id := range s.AllowedTokenIDs {
				if id < 0 || int64(id) > math.MaxInt32 {
					return "", errors.New("invalid allowed token ID")
				}
				if i > 0 {
					keys.WriteByte(',')
				}
				fmt.Fprintf(&keys, "%d", id)
			}
		}
	} else if len(s.AllowedTokenIDs) > 0 {
		return "", errors.New("allowed token IDs require logprobs")
	}
	if len(s.LogitBias) > 0 {
		ids := make([]int, 0, len(s.LogitBias))
		for id, bias := range s.LogitBias {
			if id < 0 || int64(id) > math.MaxInt32 || math.IsNaN(bias) || math.IsInf(bias, 0) || bias < -100 || bias > 100 {
				return "", errors.New("invalid logit_bias")
			}
			ids = append(ids, id)
		}
		sort.Ints(ids)
		keys.WriteString(" logit_bias=")
		for i, id := range ids {
			if i > 0 {
				keys.WriteByte(',')
			}
			fmt.Fprintf(&keys, "%d:%s", id, strconv.FormatFloat(s.LogitBias[id], 'g', -1, 64))
		}
	}
	return keys.String(), nil
}

// Generate sends a greedy text-only GEN. Cancellation sends STOP if advertised,
// then drains DONE and keeps the engine for the next request, matching the
// Python server. The process is discarded only when STOP fails, times out, or
// is unsupported. STOP sent before admission is cleared by the engine between
// requests, so it cannot leak into the next one.
func (e *Engine) Generate(ctx context.Context, ids []int, maxNew int) (Result, error) {
	return e.GenerateWithOptions(ctx, ids, maxNew, RequestOptions{})
}

func generationCommand(ids []int, maxNew int, keys string) (string, error) {
	return generationCommandWithEmbeddings(ids, maxNew, keys, "")
}

func generationCommandWithEmbeddings(ids []int, maxNew int, keys, embeddingsPath string) (string, error) {
	if maxNew < 1 || len(ids) == 0 {
		return "", errors.New("GEN requires positive maxNew and at least one token")
	}
	var command strings.Builder
	command.Grow(17 + len(keys) + len(embeddingsPath) + len(ids)*7)
	if embeddingsPath != "" {
		command.WriteString("GENI ")
	} else {
		command.WriteString("GEN ")
	}
	var number [32]byte
	command.Write(strconv.AppendInt(number[:0], int64(maxNew), 10))
	command.WriteString(keys)
	command.WriteByte(' ')
	if embeddingsPath != "" {
		command.WriteString(embeddingsPath)
		command.WriteByte(' ')
	}
	for i, id := range ids {
		if id < 0 || int64(id) > 2147483647 {
			return "", errors.New("token ID outside nonnegative int32 range")
		}
		if i > 0 {
			command.WriteByte(',')
		}
		command.Write(strconv.AppendInt(number[:0], int64(id), 10))
	}
	command.WriteByte('\n')
	return command.String(), nil
}

func generationImageCommand(ids []int, maxNew int, keys, embeddingsPath string) (string, error) {
	if strings.ContainsAny(embeddingsPath, " \t\r\n\v\f=\x00") {
		return "", errors.New("invalid embeddings path")
	}
	return generationCommandWithEmbeddings(ids, maxNew, keys, embeddingsPath)
}

// GenerateWithOptions sends GEN or GENI. ErrStopGeneration from a callback
// stops and drains the request; other callback errors discard the process once
// generation starts. Errors while queued leave the active engine untouched.
// Cancellation sends STOP and drains DONE so the process serves the next
// request, like Generate; no further callbacks run after cancellation is
// observed. A callback panic during generation discards the process and
// propagates. Callbacks must return promptly.
func (e *Engine) GenerateWithOptions(ctx context.Context, ids []int, maxNew int, opts RequestOptions) (Result, error) {
	var result Result
	keys, err := samplingKeys(opts.Sampling)
	if err != nil {
		return result, err
	}
	var command string
	if opts.EmbeddingsPath == "" {
		command, err = generationCommand(ids, maxNew, keys)
	} else {
		command, err = generationImageCommand(ids, maxNew, keys, opts.EmbeddingsPath)
	}
	if err != nil {
		return result, err
	}
	if err := acquireWithProgress(ctx, nil, e.gate, opts.OnProgress); err != nil {
		return result, err
	}
	defer func() { e.gate <- struct{}{} }()
	select {
	case <-e.exited:
		return result, ErrClosed
	default:
	}
	if len(opts.Sampling.LogitBias) > 0 && !e.SupportsLogitBias() {
		return result, errors.New("native engine does not support logit_bias without batching")
	}
	e.idle.Store(false)
	if err := e.writeCommand(ctx, command); err != nil {
		return result, err
	}
	var cancel <-chan time.Time
	var timer *time.Timer
	ctxDone := ctx.Done()
	var cancelled error
	stopping := false
	var pending *int
	var carried *record
	var lpTimer *time.Timer
	var lpWait <-chan time.Time
	var progressTimer *time.Timer
	var progressWait <-chan time.Time
	if opts.OnProgress != nil {
		progressTimer = time.NewTimer(10 * time.Second)
		progressWait = progressTimer.C
	}
	defer func() {
		if timer != nil {
			timer.Stop()
		}
		if lpTimer != nil {
			lpTimer.Stop()
		}
		if progressTimer != nil {
			progressTimer.Stop()
		}
	}()
	progress := func() error {
		if stopping || cancelled != nil || ctx.Err() != nil || opts.OnProgress == nil {
			return nil
		}
		err := func() error {
			returned := false
			defer func() {
				if !returned {
					e.abort()
				}
			}()
			err := opts.OnProgress()
			returned = true
			return err
		}()
		if errors.Is(err, ErrStopGeneration) {
			stopping = true
			if e.CanStop {
				stopCtx, stopCancel := context.WithTimeout(context.Background(), time.Second)
				err = e.writeCommand(stopCtx, "STOP\n")
				stopCancel()
				if err != nil {
					e.abort()
					return err
				}
				timer = time.NewTimer(300 * time.Second)
				cancel = timer.C
			}
			return nil
		}
		if err != nil {
			e.abort()
		}
		return err
	}
	deliverPending := func(lp *Logprob) error {
		id := *pending
		pending = nil
		lpWait = nil
		if lpTimer != nil {
			lpTimer.Stop()
		}
		if stopping || cancelled != nil || ctx.Err() != nil {
			return nil
		}
		err := func() error {
			returned := false
			defer func() {
				if !returned {
					e.abort()
				}
			}()
			var err error
			if opts.OnLogprob != nil {
				err = opts.OnLogprob(id, lp)
			}
			if err == nil && ctx.Err() == nil && opts.OnToken != nil {
				err = opts.OnToken(id)
			}
			returned = true
			return err
		}()
		if errors.Is(err, ErrStopGeneration) {
			stopping = true
			if e.CanStop && (carried == nil || !strings.HasPrefix(carried.line, "DONE ")) {
				stopCtx, stopCancel := context.WithTimeout(context.Background(), time.Second)
				err = e.writeCommand(stopCtx, "STOP\n")
				stopCancel()
				if err != nil {
					e.abort()
					return err
				}
				timer = time.NewTimer(300 * time.Second)
				cancel = timer.C
			}
			return nil
		}
		if err != nil {
			e.abort()
		}
		return err
	}
	for {
		// Cancellation from a callback must be handled before a buffered DONE.
		if ctxDone != nil && ctx.Err() != nil {
			cancelled = ctx.Err()
			ctxDone = nil
			if !e.CanStop {
				e.abort()
				return result, cancelled
			}
			if !stopping && (carried == nil || !strings.HasPrefix(carried.line, "DONE ")) {
				stopCtx, stopCancel := context.WithTimeout(context.Background(), time.Second)
				err := e.writeCommand(stopCtx, "STOP\n")
				stopCancel()
				if err != nil {
					return result, errors.Join(cancelled, err)
				}
				timer = time.NewTimer(time.Second)
				cancel = timer.C
			} else if timer != nil {
				timer.Reset(time.Second)
			}
		}
		var records <-chan record = e.records
		if carried != nil {
			ch := make(chan record, 1)
			ch <- *carried
			records = ch
		}
		select {
		case <-ctxDone:
			continue
		case <-cancel:
			e.abort()
			if cancelled == nil {
				return result, errors.New("engine did not finish after STOP")
			}
			return result, cancelled
		case <-lpWait:
			if err := deliverPending(nil); err != nil {
				return result, err
			}
			continue
		case <-progressWait:
			if err := progress(); err != nil {
				return result, err
			}
			progressTimer.Reset(10 * time.Second)
		case r, ok := <-records:
			if carried != nil {
				carried = nil
			}
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
			var lp *Logprob
			if pending != nil {
				if f[0] == "LP" {
					lp, err = parseLogprob(f)
					if err != nil {
						lp = nil
					}
				} else {
					carried = &r
				}
				if err := deliverPending(lp); err != nil {
					return result, err
				}
				continue
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
				if opts.Sampling.Logprobs != nil {
					token := int(id)
					pending = &token
					// Older engines omit LP; Python waits at most 0.5 s for it.
					if lpTimer == nil {
						lpTimer = time.NewTimer(500 * time.Millisecond)
					} else {
						lpTimer.Reset(500 * time.Millisecond)
					}
					lpWait = lpTimer.C
					continue
				}
				if !stopping && cancelled == nil && ctx.Err() == nil && opts.OnLogprob != nil {
					err := func() error {
						returned := false
						defer func() {
							if !returned {
								e.abort()
							}
						}()
						err := opts.OnLogprob(int(id), nil)
						returned = true
						return err
					}()
					if err != nil {
						if !errors.Is(err, ErrStopGeneration) {
							e.abort()
							return result, err
						}
						stopping = true
						if e.CanStop {
							stopCtx, stopCancel := context.WithTimeout(context.Background(), time.Second)
							err = e.writeCommand(stopCtx, "STOP\n")
							stopCancel()
							if err != nil {
								return result, err
							}
							timer = time.NewTimer(300 * time.Second)
							cancel = timer.C
						}
					}
				}
				if !stopping && cancelled == nil && ctx.Err() == nil && opts.OnToken != nil {
					err := func() error {
						returned := false
						defer func() {
							if !returned {
								e.abort()
							}
						}()
						err := opts.OnToken(int(id))
						returned = true
						return err
					}()
					if errors.Is(err, ErrStopGeneration) {
						stopping = true
						if e.CanStop {
							stopCtx, stopCancel := context.WithTimeout(context.Background(), time.Second)
							err = e.writeCommand(stopCtx, "STOP\n")
							stopCancel()
							if err != nil {
								e.abort()
								return result, err
							}
							// Match the Python server's default silence allowance:
							// a slow native decode step can take much longer than 10s.
							timer = time.NewTimer(300 * time.Second)
							cancel = timer.C
						}
					} else if err != nil {
						e.abort()
						return result, err
					}
				}
			case "PP", "RESUME", "REUSED":
				if f[0] == "PP" {
					if !stopping && cancelled == nil && ctx.Err() == nil && opts.OnPromptProgress != nil && len(f) >= 3 {
						current, err1 := strconv.Atoi(f[1])
						total, err2 := strconv.Atoi(f[2])
						if err1 == nil && err2 == nil && current >= 0 && total > 0 && current <= total {
							func() {
								returned := false
								defer func() {
									if !returned {
										e.abort()
									}
								}()
								opts.OnPromptProgress(current, total)
								returned = true
							}()
						}
					}
					if err := progress(); err != nil {
						return result, err
					}
				}
			case "LP":
				if opts.Sampling.Logprobs == nil {
					e.abort()
					return result, fmt.Errorf("unsupported engine record %q", r.line)
				}
				// An LP that arrived after the compatibility timeout is stale.
			case "ERR":
				e.abort()
				return result, EngineError(strings.TrimSpace(strings.TrimPrefix(r.line, "ERR")))
			case "DONE":
				if err := parseDone(f, &result); err != nil {
					e.abort()
					return result, err
				}
				if cancelled != nil {
					// A STOP-acknowledged cancel leaves the drained engine for
					// the next request, like Python's STOP-and-drain. The
					// engine clears a between-request STOP, so no stale stop
					// leaks into it. Keep RSS flat: no extra state or buffers.
					return result, cancelled
				}
				if err := ctx.Err(); err != nil {
					e.abort()
					return result, err
				}
				if stopping {
					result.Finish = "stop"
				}
				return result, nil
			default:
				e.abort()
				return result, fmt.Errorf("unsupported engine record %q", r.line)
			}
		}
	}
}

func parseLogprob(f []string) (*Logprob, error) {
	if len(f) < 2 || f[0] != "LP" {
		return nil, errors.New("malformed LP")
	}
	// The native engine uses these values when logits are unavailable.
	if len(f) == 2 && (strings.EqualFold(f[1], "nan") || strings.EqualFold(f[1], "-nan(ind)")) {
		return nil, nil
	}
	parseValue := func(s string) (float64, error) {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, errors.New("malformed LP value")
		}
		return v, nil
	}
	value, err := parseValue(f[1])
	if err != nil {
		return nil, err
	}
	lp := &Logprob{Value: value}
	for _, item := range f[2:] {
		idText, valueText, ok := strings.Cut(item, ":")
		id, idErr := strconv.ParseInt(idText, 10, 32)
		v, valueErr := parseValue(valueText)
		if !ok || idErr != nil || id < 0 || valueErr != nil {
			return nil, errors.New("malformed LP alternative")
		}
		lp.Alternatives = append(lp.Alternatives, LogprobAlternative{TokenID: int(id), Value: v})
	}
	return lp, nil
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
