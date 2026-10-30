package vision

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func lazyResult(t *testing.T, ch <-chan error) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for lazy reader")
		return nil
	}
}

func TestLazyOutputSurvivesRepeatedProcesses(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	env := []string{"VISION_CHILD=1", "VISION_MODE=ok"}
	l := NewLazy(os.Args[0], childArgs("ok"), env)
	env[1] = "VISION_MODE=failure"
	entries, err := os.ReadDir(tmp)
	if err != nil || len(entries) != 0 {
		t.Fatalf("constructor created files: %v, %v", entries, err)
	}
	for i := 0; i < 2; i++ {
		path, tokens, err := l.Encode(context.Background(), []byte("image"))
		if err != nil {
			t.Fatal(err)
		}
		defer os.Remove(path)
		data, err := os.ReadFile(path)
		if err != nil || tokens != 2 || len(data) != 36 || binary.LittleEndian.Uint32(data) != 0x31455653 {
			t.Fatalf("invalid output: tokens=%d data=%x err=%v", tokens, data, err)
		}
		dirs, err := filepath.Glob(filepath.Join(tmp, "strata-vision-*"))
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range dirs {
			info, err := os.Stat(p)
			if err != nil || info.IsDir() {
				t.Fatalf("reader directory remained: %s, %v", p, err)
			}
		}
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.Encode(context.Background(), []byte("image")); err == nil {
		t.Fatal("encode succeeded after close")
	}
}

func TestLazyCancellationAndCloseReapChild(t *testing.T) {
	for _, closeReader := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "close"}[closeReader], func(t *testing.T) {
			t.Setenv("VISION_CHILD", "1")
			t.Setenv("VISION_MODE", "silent-encode")
			tmp := t.TempDir()
			t.Setenv("TMPDIR", tmp)
			listener := stallListener(t)
			l := NewLazy(os.Args[0], childArgs("silent-encode"), nil)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result := make(chan error, 1)
			go func() { _, _, err := l.Encode(ctx, []byte("image")); result <- err }()
			conn := waitForStall(t, listener)
			if closeReader {
				closed := make(chan error, 1)
				go func() { closed <- l.Close() }()
				if err := lazyResult(t, closed); err != nil {
					t.Fatal(err)
				}
			} else {
				cancel()
			}
			if err := lazyResult(t, result); !errors.Is(err, context.Canceled) {
				t.Fatalf("encode cancellation: %v", err)
			}
			_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
			var b [1]byte
			if n, err := conn.Read(b[:]); n != 0 || err == nil {
				t.Fatalf("child connection remained open: n=%d err=%v", n, err)
			}
			entries, err := os.ReadDir(tmp)
			if err != nil || len(entries) != 0 {
				t.Fatalf("temporary files remained: %v, %v", entries, err)
			}
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
