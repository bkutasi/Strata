//go:build unix

package main

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestOwnerEscalationFixture(t *testing.T) {
	if os.Getenv("STRATA_OWNER_ESCALATION") != "1" {
		return
	}
	term := make(chan os.Signal, 1)
	signal.Notify(term, syscall.SIGTERM)
	deadline := make(chan time.Time)
	go func() {
		<-term
		fmt.Fprintln(os.Stderr, "owner-term-received")
		close(deadline)
	}()
	stopOwnerGroup(syscall.Getpgrp(), deadline, func() {})
	os.Exit(2)
}

func TestOwnerGuardEscalatesWhenGraceExpires(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestOwnerEscalationFixture$")
	cmd.Env = append(os.Environ(), "STRATA_OWNER_ESCALATION=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	out, err := cmd.CombinedOutput()
	exit, ok := err.(*exec.ExitError)
	if !ok || !exit.ProcessState.Sys().(syscall.WaitStatus).Signaled() ||
		exit.ProcessState.Sys().(syscall.WaitStatus).Signal() != syscall.SIGKILL ||
		!strings.Contains(string(out), "owner-term-received") {
		t.Fatalf("expected TERM then KILL, got %v: %s", err, out)
	}
}

func TestOwnerPipeStopsServerAndEngine(t *testing.T) {
	for _, tc := range []struct {
		name     string
		starting bool
		signal   bool
	}{
		{"serving owner loss", false, false},
		{"startup owner loss", true, false},
		{"signal with owner alive", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reader, writer, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()
			defer writer.Close()
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(exe, "-test.run=^TestLifecycleFixture$", "--", "server",
				"-listen", "127.0.0.1:0", "-model", "test", "-pack", lifecyclePack(t),
				"-engine", exe, "-engine-arg=-test.run=^TestLifecycleFixture$",
				"-engine-arg=--", "-engine-arg=engine", "-owner-fd", "3")
			startingValue := "0"
			if tc.starting {
				startingValue = "1"
			}
			cmd.Env = append(os.Environ(), "STRATA_LIFECYCLE_FIXTURE=1", "STRATA_LIFECYCLE_STARTING="+startingValue)
			cmd.ExtraFiles = []*os.File{reader}
			cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
			stderr, err := cmd.StderrPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = cmd.Process.Kill() })
			if err := reader.Close(); err != nil {
				t.Fatal(err)
			}
			lines := make(chan string, 64)
			go func() {
				defer close(lines)
				scanner := bufio.NewScanner(stderr)
				for scanner.Scan() {
					lines <- scanner.Text()
				}
			}()
			waitLine := func(prefix string) string {
				t.Helper()
				timer := time.NewTimer(20 * time.Second)
				defer timer.Stop()
				for {
					select {
					case line, ok := <-lines:
						if !ok {
							t.Fatalf("server exited before %s", prefix)
						}
						if strings.Contains(line, prefix) {
							return line
						}
					case <-timer.C:
						t.Fatalf("timed out waiting for %s", prefix)
					}
				}
			}
			line := waitLine("fixture-engine-pid ")
			pid, err := strconv.Atoi(strings.TrimPrefix(line, "fixture-engine-pid "))
			if err != nil {
				t.Fatal(err)
			}
			engineReaped := false
			t.Cleanup(func() {
				if !engineReaped {
					_ = syscall.Kill(pid, syscall.SIGKILL)
				}
			})
			if _, err := writer.Write([]byte("ping")); err != nil {
				t.Fatal(err)
			}
			if !tc.starting {
				line := waitLine("experimental Go server listening on ")
				addr := strings.SplitN(line, "experimental Go server listening on ", 2)[1]
				client := &http.Client{Timeout: 5 * time.Second}
				response, err := client.Get("http://" + addr + "/health")
				if err != nil {
					t.Fatal(err)
				}
				_ = response.Body.Close()
				if response.StatusCode != http.StatusOK {
					t.Fatalf("health status %d", response.StatusCode)
				}
			}
			if tc.signal {
				if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := writer.Close(); err != nil {
					t.Fatal(err)
				}
			}
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			select {
			case err := <-done:
				if err != nil {
					if e, ok := err.(*exec.ExitError); !ok || e.ProcessState.Sys().(syscall.WaitStatus).Signaled() {
						t.Fatalf("server exit: %v", err)
					}
				}
			case <-time.After(20 * time.Second):
				t.Fatal("server did not exit after its owner pipe closed")
			}
			if err := syscall.Kill(pid, 0); err != syscall.ESRCH {
				t.Fatalf("engine %d was not reaped: %v", pid, err)
			}
			engineReaped = true
		})
	}
}

func TestOwnerPipeRejectsUnsafeDescriptors(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, want string
		file       func() (*os.File, func())
		group      bool
	}{
		{"regular file", "--owner-fd must be an inherited pipe", func() (*os.File, func()) {
			f, err := os.CreateTemp(t.TempDir(), "owner")
			if err != nil {
				t.Fatal(err)
			}
			return f, func() { _ = f.Close() }
		}, true},
		{"shared process group", "--owner-fd requires a dedicated POSIX server process group", func() (*os.File, func()) {
			r, w, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			return r, func() { _ = r.Close(); _ = w.Close() }
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			file, closeFile := tc.file()
			defer closeFile()
			cmd := exec.Command(exe, "-test.run=^TestLifecycleFixture$", "--", "server",
				"-model", "test", "-pack", t.TempDir(), "-engine", exe, "-owner-fd", "3")
			cmd.Env = append(os.Environ(), "STRATA_LIFECYCLE_FIXTURE=1")
			cmd.ExtraFiles = []*os.File{file}
			cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: tc.group}
			output, err := cmd.CombinedOutput()
			if err == nil || !strings.Contains(string(output), tc.want) {
				t.Fatalf("expected owner validation error, got %v: %s", err, output)
			}
		})
	}
}
