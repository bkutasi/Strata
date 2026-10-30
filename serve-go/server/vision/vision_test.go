package vision

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVisionChild(t *testing.T) {
	if os.Getenv("VISION_CHILD") != "1" {
		return
	}
	switch os.Getenv("VISION_MODE") {
	case "env-restart":
		if os.Getenv("VISION_DEVICE") != "selected-gpu" {
			fmt.Println("invalid environment")
			return
		}
		fmt.Println("READY 2")
	case "silent":
		stallChild(t)
		return
	case "badready":
		fmt.Println("READY nope")
		return
	case "huge-ready":
		fmt.Println("READY 4294967295")
	case "long-ready":
		fmt.Println(strings.Repeat("x", maxProtocolLine+1))
		return
	default:
		fmt.Println("READY 2")
	}
	if os.Getenv("VISION_MODE") == "exit" {
		os.Exit(0)
	}
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		switch {
		case sc.Text() == "QUIT":
			if os.Getenv("VISION_MODE") == "ignore-quit" {
				stallChild(t)
			}
			return
		case strings.HasPrefix(sc.Text(), "ENC "):
			var in, out string
			_, _ = fmt.Sscanf(sc.Text(), "ENC %s %s", &in, &out)
			if os.Getenv("VISION_MODE") == "restart" || os.Getenv("VISION_MODE") == "env-restart" {
				marker := os.Getenv("VISION_MARKER")
				if _, err := os.Stat(marker); os.IsNotExist(err) {
					_ = os.WriteFile(marker, []byte("failed"), 0600)
					fmt.Println("unexpected response")
					continue
				}
			}
			if os.Getenv("VISION_MODE") == "failure" {
				fmt.Println("ERR synthetic failure")
				continue
			}
			if os.Getenv("VISION_MODE") == "failure-once" && in == "input-1.img" {
				_ = os.WriteFile(out, []byte("partial"), 0600)
				fmt.Println("ERR synthetic failure")
				continue
			}
			if os.Getenv("VISION_MODE") == "silent-encode" {
				stallChild(t)
				return
			}
			if os.Getenv("VISION_MODE") == "huge-ready" {
				data := make([]byte, 20)
				for i, v := range []uint32{0x31455653, ^uint32(0), ^uint32(0), 1, ^uint32(0)} {
					binary.LittleEndian.PutUint32(data[i*4:], v)
				}
				_ = os.WriteFile(out, data, 0600)
				fmt.Println("OK 4294967295 4294967295 1 0")
				continue
			}
			if os.Getenv("VISION_MODE") == "oversized-grid" {
				data := make([]byte, 20+2*2*4)
				for i, v := range []uint32{0x31455653, 2, 2, 2, 2} {
					binary.LittleEndian.PutUint32(data[i*4:], v)
				}
				_ = os.WriteFile(out, data, 0600)
				fmt.Println("OK 2 2 2 0")
				continue
			}
			b, _ := os.ReadFile(in)
			if len(b) == 0 {
				fmt.Println("ERR missing input")
				continue
			}
			data := make([]byte, 20+2*2*4)
			for i, v := range []uint32{0x31455653, 2, 1, 2, 2} {
				binary.LittleEndian.PutUint32(data[i*4:], v)
			}
			_ = os.WriteFile(out, data, 0600)
			fmt.Println("OK 2 1 2 0")
		}
	}
}

func stallChild(t *testing.T) {
	t.Helper()
	conn, err := net.Dial("tcp", os.Getenv("VISION_SIGNAL"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	var b [1]byte
	_, _ = conn.Read(b[:])
}

func stallListener(t *testing.T) *net.TCPListener {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	t.Setenv("VISION_SIGNAL", listener.Addr().String())
	return listener.(*net.TCPListener)
}

func waitForStall(t *testing.T, listener *net.TCPListener) net.Conn {
	t.Helper()
	_ = listener.SetDeadline(time.Now().Add(5 * time.Second))
	conn, err := listener.Accept()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func childArgs(mode string) []string {
	return []string{"-test.run=^TestVisionChild$", "--"}
}

func TestStartEncodeClose(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "ok")
	r, err := Start(context.Background(), os.Args[0], childArgs("ok"))
	if err != nil {
		t.Fatal(err)
	}
	dir := r.dir
	path, n, err := r.Encode(context.Background(), []byte("image"))
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 || path != filepath.Join(dir, "output-1.sve") {
		t.Fatalf("path=%q tokens=%d", path, n)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("output missing after encode: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("directory remained: %v", err)
	}
}

func TestStartAndEncodeFailuresCleanup(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "badready")
	if r, err := Start(context.Background(), os.Args[0], childArgs("badready")); err == nil {
		_ = r.Close()
		t.Fatal("accepted invalid READY")
	}
	t.Setenv("VISION_MODE", "failure")
	r, err := Start(context.Background(), os.Args[0], childArgs("failure"))
	if err != nil {
		t.Fatal(err)
	}
	dir := r.dir
	if _, _, err := r.Encode(context.Background(), []byte("image")); err == nil {
		t.Fatal("accepted ERR")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("directory remained: %v", err)
	}
}

func TestStartupCancellation(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "silent")
	listener := stallListener(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		r, err := Start(ctx, os.Args[0], childArgs("silent"))
		if err == nil {
			_ = r.Close()
		}
		result <- err
	}()
	waitForStall(t, listener)
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("startup cancellation: %v", err)
	}
}

func TestEncodeCancellation(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "silent-encode")
	listener := stallListener(t)
	r, err := Start(context.Background(), os.Args[0], childArgs("silent-encode"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() { _, _, err := r.Encode(ctx, []byte("image")); result <- err }()
	waitForStall(t, listener)
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("encode cancellation: %v", err)
	}
	if _, err := os.Stat(r.dir); !os.IsNotExist(err) {
		t.Fatalf("directory remained after cancelled encode: %v", err)
	}
}

func TestProcessExitCleanup(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "exit")
	r, err := Start(context.Background(), os.Args[0], childArgs("exit"))
	if err != nil {
		t.Fatal(err)
	}
	dir := r.dir
	<-r.done
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("directory remained after process exit: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.out.Read(make([]byte, 1)); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("stdout pipe not closed: %v", err)
	}
}

func TestReadyAfterProcessWait(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "exit")
	out, childOut, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	cmd := exec.Command(os.Args[0], childArgs("exit")...)
	cmd.Stdout = childOut
	if err := cmd.Start(); err != nil {
		_ = childOut.Close()
		t.Fatal(err)
	}
	_ = childOut.Close()
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	r := &Reader{stdout: bufio.NewReaderSize(out, maxProtocolLine)}
	line, err := r.readLine(context.Background())
	if err != nil || line != "READY 2\n" {
		t.Fatalf("READY after Wait: %q, %v", line, err)
	}
	if _, err := r.readLine(context.Background()); err != io.EOF {
		t.Fatalf("expected EOF after READY, got %v", err)
	}
}

func TestEncodeErrorThenSuccess(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "failure-once")
	r, err := Start(context.Background(), os.Args[0], childArgs("failure-once"))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, _, err := r.Encode(context.Background(), []byte("bad")); err == nil || !strings.Contains(err.Error(), "synthetic failure") {
		t.Fatalf("expected image error, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "output-1.sve")); !os.IsNotExist(err) {
		t.Fatalf("failed output remained: %v", err)
	}
	path, tokens, err := r.Encode(context.Background(), []byte("good"))
	if err != nil || tokens != 2 {
		t.Fatalf("encode after ERR: tokens=%d, err=%v", tokens, err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}

func TestEncodeRestartsAfterProtocolErrorAndCloseIsFinal(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "restart")
	t.Setenv("VISION_MARKER", filepath.Join(t.TempDir(), "marker"))
	r, err := Start(context.Background(), os.Args[0], childArgs("restart"))
	if err != nil {
		t.Fatal(err)
	}
	oldDir, oldDone := r.dir, r.done
	if _, _, err := r.Encode(context.Background(), []byte("bad")); err == nil {
		t.Fatal("accepted malformed response")
	}
	select {
	case <-oldDone:
	default:
		t.Fatal("old process not reaped")
	}
	path, tokens, err := r.Encode(context.Background(), []byte("good"))
	if err != nil || tokens != 2 {
		t.Fatalf("encode after restart: tokens=%d, err=%v", tokens, err)
	}
	if r.dir == oldDir {
		t.Fatal("restart reused old work directory")
	}
	if _, err := os.Stat(oldDir); !os.IsNotExist(err) {
		t.Fatalf("old directory remained: %v", err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.Encode(context.Background(), []byte("good")); err == nil {
		t.Fatal("encode succeeded after Close")
	}
}

func TestStartWithEnvPreservesEnvironmentOnRestart(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "marker")
	env := []string{"VISION_CHILD=1", "VISION_MODE=env-restart", "VISION_DEVICE=selected-gpu", "VISION_MARKER=" + marker}
	r, err := StartWithEnv(context.Background(), os.Args[0], childArgs("env-restart"), env)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	env[2] = "VISION_DEVICE=wrong-gpu"
	if _, _, err := r.Encode(context.Background(), []byte("image")); err == nil {
		t.Fatal("accepted malformed response")
	}
	path, tokens, err := r.Encode(context.Background(), []byte("image"))
	if err != nil || tokens != 2 {
		t.Fatalf("restart with saved environment: tokens=%d, err=%v", tokens, err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}

func TestEncodeRestartsAfterCancellation(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "silent-encode")
	listener := stallListener(t)
	r, err := Start(context.Background(), os.Args[0], childArgs("silent-encode"))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	oldDir := r.dir
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() { _, _, err := r.Encode(ctx, []byte("image")); result <- err }()
	waitForStall(t, listener)
	cancel()
	if err := <-result; err == nil {
		t.Fatal("expected cancelled encode")
	}
	t.Setenv("VISION_MODE", "ok")
	path, _, err := r.Encode(context.Background(), []byte("image"))
	if err != nil {
		t.Fatal(err)
	}
	if r.dir == oldDir {
		t.Fatal("restart reused old work directory")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
}

func TestStartupContextDoesNotOwnProcess(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "ok")
	ctx, cancel := context.WithCancel(context.Background())
	r, err := Start(ctx, os.Args[0], childArgs("ok"))
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	path, _, err := r.Encode(context.Background(), []byte("image"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCloseKillsIgnoringQuit(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "ignore-quit")
	listener := stallListener(t)
	r, err := Start(context.Background(), os.Args[0], childArgs("ignore-quit"))
	if err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	go func() { result <- r.Close() }()
	waitForStall(t, listener)
	select {
	case <-r.done:
		t.Fatal("child exited while ignoring QUIT")
	default:
	}
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	if r.cmd.ProcessState.Success() {
		t.Fatal("child exited successfully instead of being killed")
	}
	select {
	case <-r.done:
	default:
		t.Fatal("process not reaped")
	}
	if _, err := os.Stat(r.dir); !os.IsNotExist(err) {
		t.Fatalf("directory remained: %v", err)
	}
}

func TestConcurrentEncodeCancellation(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "silent-encode")
	listener := stallListener(t)
	r, err := Start(context.Background(), os.Args[0], childArgs("silent-encode"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)
	go func() { _, _, err := r.Encode(ctx, []byte("image")); first <- err }()
	waitForStall(t, listener)
	waiting, stop := context.WithCancel(context.Background())
	second := make(chan error, 1)
	started := make(chan struct{})
	go func() {
		close(started)
		_, _, err := r.Encode(waiting, []byte("image"))
		second <- err
	}()
	<-started
	stop()
	if err := <-second; err != context.Canceled {
		t.Fatalf("waiting encode: %v", err)
	}
	cancel()
	if err := <-first; err == nil {
		t.Fatal("active encode did not cancel")
	}
	select {
	case <-r.done:
	default:
		t.Fatal("cancelled process not reaped")
	}
}

func TestOversizedProtocolAndHeader(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "long-ready")
	if r, err := Start(context.Background(), os.Args[0], childArgs("long-ready")); err == nil {
		_ = r.Close()
		t.Fatal("accepted oversized protocol line")
	}
	t.Setenv("VISION_MODE", "huge-ready")
	r, err := Start(context.Background(), os.Args[0], childArgs("huge-ready"))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, _, err := r.Encode(context.Background(), []byte("image")); err == nil || !strings.Contains(err.Error(), "oversized") {
		t.Fatalf("huge header: %v", err)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "output-1.sve")); !os.IsNotExist(err) {
		t.Fatalf("invalid output remained: %v", err)
	}
}

func TestRejectGridLargerThanTokenCount(t *testing.T) {
	t.Setenv("VISION_CHILD", "1")
	t.Setenv("VISION_MODE", "oversized-grid")
	r, err := Start(context.Background(), os.Args[0], childArgs("oversized-grid"))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, _, err := r.Encode(context.Background(), []byte("image")); err == nil || !strings.Contains(err.Error(), "invalid SVE1 header") {
		t.Fatalf("expected invalid header, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "output-1.sve")); !os.IsNotExist(err) {
		t.Fatalf("invalid output remained: %v", err)
	}
}
