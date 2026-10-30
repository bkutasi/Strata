//go:build unix

package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The test binary supplies both server and native-engine subprocesses.
func TestLifecycleFixture(t *testing.T) {
	if os.Getenv("STRATA_LIFECYCLE_FIXTURE") != "1" {
		return
	}
	i := 0
	for i < len(os.Args) && os.Args[i] != "--" {
		i++
	}
	if i+1 >= len(os.Args) {
		os.Exit(2)
	}
	if os.Args[i+1] == "engine" {
		fmt.Fprintf(os.Stderr, "fixture-engine-pid %d\n", os.Getpid())
		if os.Getenv("STRATA_LIFECYCLE_STARTING") == "1" {
			ch := make(chan os.Signal, 1)
			signal.Notify(ch, syscall.SIGUSR1)
			<-ch
			os.Exit(3)
		}
		fmt.Println("READY 4096 stop")
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			if scanner.Text() == "QUIT" {
				fmt.Fprintln(os.Stderr, "fixture-engine-quit")
				os.Exit(0)
			}
		}
		os.Exit(4)
	}
	flag.CommandLine = flag.NewFlagSet("server", flag.ContinueOnError)
	os.Args = append([]string{"server"}, os.Args[i+2:]...)
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

func lifecyclePack(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	vocab := make(map[string]int)
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = rune(256 + n)
			n++
		}
		vocab[string(r)] = b
	}
	vocab["ab"] = 256
	writeJSON := func(name string, v any) {
		t.Helper()
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON("tokenizer.json", map[string]string{"pre": "qwen35", "model": "gpt2"})
	writeJSON("vocab.json", vocab)
	writeJSON("token_type.json", make([]int, len(vocab)))
	for name, data := range map[string]string{"merges.txt": "a b", "chat_template.jinja": "{{ messages[0].content }}"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestSIGTERMStopsServerAndEngine(t *testing.T) {
	for _, starting := range []bool{false, true} {
		t.Run(strconv.FormatBool(starting), func(t *testing.T) {
			pack := lifecyclePack(t)
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(exe, "-test.run=^TestLifecycleFixture$", "--", "server",
				"-listen", "127.0.0.1:0", "-model", "test", "-pack", pack, "-engine", exe,
				"-engine-arg=-test.run=^TestLifecycleFixture$", "-engine-arg=--", "-engine-arg=engine")
			startingValue := "0"
			if starting {
				startingValue = "1"
			}
			cmd.Env = append(os.Environ(), "STRATA_LIFECYCLE_FIXTURE=1", "STRATA_LIFECYCLE_STARTING="+startingValue)
			pipe, err := cmd.StderrPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = cmd.Process.Kill() })
			lines := make(chan string, 64)
			go func() {
				defer close(lines)
				scanner := bufio.NewScanner(pipe)
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
							t.Fatalf("process exited before %s", prefix)
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
			if !starting {
				line := waitLine("experimental Go server listening on ")
				addr := strings.SplitN(line, "experimental Go server listening on ", 2)[1]
				client := &http.Client{Timeout: 5 * time.Second}
				response, err := client.Get("http://" + addr + "/health")
				if err != nil {
					t.Fatal(err)
				}
				if err := response.Body.Close(); err != nil {
					t.Fatal(err)
				}
				if response.StatusCode != http.StatusOK {
					t.Fatalf("health status %d", response.StatusCode)
				}
			}
			if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
				t.Fatal(err)
			}
			if !starting {
				waitLine("fixture-engine-quit")
			}
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("server exit: %v", err)
				}
			case <-time.After(20 * time.Second):
				t.Fatal("server did not exit")
			}
			if err := syscall.Kill(pid, 0); err != syscall.ESRCH {
				t.Fatalf("engine %d was not reaped: %v", pid, err)
			}
			engineReaped = true
		})
	}
}
