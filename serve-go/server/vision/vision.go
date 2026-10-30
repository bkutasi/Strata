package vision

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const sveHeaderSize = 20
const maxProtocolLine = 4096
const closeTimeout = 500 * time.Millisecond

type stderrBuffer struct {
	mu sync.Mutex
	b  []byte
}

func (s *stderrBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(p)
	s.b = append(s.b, p...)
	if len(s.b) > 8192 {
		s.b = append([]byte(nil), s.b[len(s.b)-8192:]...)
	}
	return n, nil
}

func (s *stderrBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.TrimSpace(string(s.b))
}

type Reader struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	out    *os.File
	dir    string
	embd   int
	lock   chan struct{}
	stderr *stderrBuffer
	closed bool
	dead   bool
	exe    string
	args   []string
	env    []string
	done   chan struct{}
	seq    uint64
}

// Start launches the resident strata-vision process and waits for its READY line.
func Start(ctx context.Context, exe string, args []string) (*Reader, error) {
	return StartWithEnv(ctx, exe, args, nil)
}

// StartWithEnv launches the reader with env as its complete environment.
// A nil env inherits the current process environment.
func StartWithEnv(ctx context.Context, exe string, args, env []string) (*Reader, error) {
	var savedEnv []string
	if env != nil {
		savedEnv = append([]string{}, env...)
	}
	dir, err := os.MkdirTemp("", "strata-vision-")
	if err != nil {
		return nil, err
	}
	if strings.ContainsAny(dir, " \t\r\n=") {
		_ = os.Remove(dir)
		return nil, fmt.Errorf("vision temporary path contains characters unsafe for GENI: %q", dir)
	}
	if err := ctx.Err(); err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	cmd := exec.Command(exe, args...)
	cmd.Env = savedEnv
	cmd.Dir = dir
	stderr := &stderrBuffer{}
	cmd.Stderr = stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	out, childOut, err := os.Pipe()
	if err != nil {
		_ = stdin.Close()
		_ = os.RemoveAll(dir)
		return nil, err
	}
	cmd.Stdout = childOut
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = out.Close()
		_ = childOut.Close()
		_ = os.RemoveAll(dir)
		return nil, err
	}
	_ = childOut.Close()
	r := &Reader{cmd: cmd, stdin: stdin, stdout: bufio.NewReaderSize(out, maxProtocolLine), out: out, dir: dir, done: make(chan struct{}), lock: make(chan struct{}, 1), stderr: stderr, exe: exe, args: append([]string(nil), args...), env: savedEnv}
	r.lock <- struct{}{}
	go func(done chan struct{}, dir string) {
		_ = cmd.Wait()
		_ = os.RemoveAll(dir)
		close(done)
	}(r.done, dir)
	line, err := r.readLine(ctx)
	if err != nil {
		r.abort()
		return nil, r.diagnostic(fmt.Errorf("vision startup: %w", err))
	}
	fields := strings.Fields(line)
	if len(fields) != 2 || fields[0] != "READY" {
		r.abort()
		return nil, r.diagnostic(fmt.Errorf("vision startup: invalid response %q", strings.TrimSpace(line)))
	}
	n, err := strconv.Atoi(fields[1])
	if err != nil || n <= 0 {
		r.abort()
		return nil, r.diagnostic(fmt.Errorf("vision startup: invalid embedding dimension %q", fields[1]))
	}
	r.embd = n
	return r, nil
}

func (r *Reader) readLine(ctx context.Context) (string, error) {
	type result struct {
		line string
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		line, err := r.stdout.ReadSlice('\n')
		ch <- result{string(line), err}
	}()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case x := <-ch:
		return x.line, x.err
	}
}

// Encode writes image bytes and returns the SVE file path and token count.
// The caller must remove each returned output file when it is no longer needed.
func (r *Reader) Encode(ctx context.Context, image []byte) (path string, tokens int, err error) {
	select {
	case <-ctx.Done():
		return "", 0, ctx.Err()
	case <-r.lock:
	}
	defer func() { r.lock <- struct{}{} }()
	if r.closed {
		return "", 0, fmt.Errorf("vision reader is closed")
	}
	if err := ctx.Err(); err != nil {
		return "", 0, err
	}
	select {
	case <-r.done:
		r.dead = true
	default:
	}
	if r.dead {
		_ = r.out.Close()
		next, startErr := StartWithEnv(ctx, r.exe, r.args, r.env)
		if startErr != nil {
			return "", 0, fmt.Errorf("vision restart: %w", startErr)
		}
		r.cmd, r.stdin, r.stdout, r.out = next.cmd, next.stdin, next.stdout, next.out
		r.dir, r.embd, r.done, r.stderr = next.dir, next.embd, next.done, next.stderr
		r.dead = false
	}
	r.seq++
	inName, outName := fmt.Sprintf("input-%d.img", r.seq), fmt.Sprintf("output-%d.sve", r.seq)
	inPath, outPath := filepath.Join(r.dir, inName), filepath.Join(r.dir, outName)
	defer func() {
		if err != nil {
			_ = os.Remove(outPath)
		}
	}()
	if err := os.WriteFile(inPath, image, 0600); err != nil {
		return "", 0, err
	}
	defer os.Remove(inPath)
	if err := r.write(ctx, fmt.Sprintf("ENC %s %s\n", inName, outName)); err != nil {
		r.abort()
		return "", 0, r.diagnostic(fmt.Errorf("vision command: %w", err))
	}
	line, err := r.readLine(ctx)
	if err != nil {
		r.abort()
		return "", 0, r.diagnostic(fmt.Errorf("vision encode: %w", err))
	}
	fields := strings.Fields(line)
	if len(fields) >= 2 && fields[0] == "ERR" {
		return "", 0, r.diagnostic(fmt.Errorf("vision encode failed: %q", strings.TrimSpace(line)))
	}
	if len(fields) != 5 || fields[0] != "OK" {
		r.abort()
		return "", 0, r.diagnostic(fmt.Errorf("vision encode failed: %q", strings.TrimSpace(line)))
	}
	tokens, err = strconv.Atoi(fields[1])
	if err != nil || tokens <= 0 {
		r.abort()
		return "", 0, fmt.Errorf("vision returned invalid token count %q", fields[1])
	}
	nxReply, err := strconv.ParseUint(fields[2], 10, 32)
	if err != nil || nxReply == 0 {
		r.abort()
		return "", 0, fmt.Errorf("vision returned invalid grid width %q", fields[2])
	}
	nyReply, err := strconv.ParseUint(fields[3], 10, 32)
	if err != nil || nyReply == 0 {
		r.abort()
		return "", 0, fmt.Errorf("vision returned invalid grid height %q", fields[3])
	}
	f, err := os.Open(outPath)
	if err != nil {
		return "", 0, fmt.Errorf("vision output: %w", err)
	}
	h := make([]byte, sveHeaderSize)
	_, readErr := io.ReadFull(f, h)
	info, statErr := f.Stat()
	closeErr := f.Close()
	if readErr != nil {
		return "", 0, fmt.Errorf("vision output header: %w", readErr)
	}
	if statErr != nil {
		return "", 0, statErr
	}
	if closeErr != nil {
		return "", 0, closeErr
	}
	n, nx, ny, embd := uint64(binary.LittleEndian.Uint32(h[4:])), uint64(binary.LittleEndian.Uint32(h[8:])), uint64(binary.LittleEndian.Uint32(h[12:])), uint64(binary.LittleEndian.Uint32(h[16:]))
	if binary.LittleEndian.Uint32(h) != 0x31455653 || n != uint64(tokens) || nx != nxReply || ny != nyReply || embd != uint64(r.embd) || nx*ny != n {
		return "", 0, fmt.Errorf("vision output has invalid SVE1 header")
	}
	if embd != 0 && n > (uint64(^uint64(0)>>1)-sveHeaderSize)/4/embd {
		return "", 0, fmt.Errorf("vision output has oversized SVE1 header")
	}
	expected := uint64(sveHeaderSize) + n*embd*4
	if expected > uint64(^uint64(0)>>1) || uint64(info.Size()) != expected {
		return "", 0, fmt.Errorf("vision output length %d, expected %d", info.Size(), expected)
	}
	return outPath, tokens, nil
}

// Close stops the process and removes its private work directory.
func (r *Reader) Close() error {
	<-r.lock
	defer func() { r.lock <- struct{}{} }()
	if r.closed {
		return nil
	}
	r.closed = true
	if r.dead {
		return nil
	}
	select {
	case <-r.done:
		return errorsJoin(r.out.Close(), os.RemoveAll(r.dir))
	default:
	}
	ctx, cancel := context.WithTimeout(context.Background(), closeTimeout)
	defer cancel()
	writeErr := r.write(ctx, "QUIT\n")
	_ = r.stdin.Close()
	select {
	case <-r.done:
	case <-ctx.Done():
		_ = r.cmd.Process.Kill()
		<-r.done
	}
	if ctx.Err() != nil {
		writeErr = nil
	}
	return errorsJoin(writeErr, r.out.Close(), os.RemoveAll(r.dir))
}

func (r *Reader) abort() {
	if !r.dead {
		r.dead = true
		_ = r.cmd.Process.Kill()
		<-r.done
		_ = r.out.Close()
		_ = os.RemoveAll(r.dir)
	}
}

func (r *Reader) write(ctx context.Context, command string) error {
	ch := make(chan error, 1)
	go func() {
		_, err := io.WriteString(r.stdin, command)
		ch <- err
	}()
	select {
	case err := <-ch:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Reader) diagnostic(err error) error {
	if s := r.stderr.String(); s != "" {
		return fmt.Errorf("%w (stderr: %s)", err, s)
	}
	return err
}

func errorsJoin(errs ...error) error {
	var out error
	for _, err := range errs {
		if err != nil {
			if out == nil {
				out = err
			} else {
				out = fmt.Errorf("%v; %w", out, err)
			}
		}
	}
	return out
}
