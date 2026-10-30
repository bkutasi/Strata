//go:build unix

package main

import (
	"bufio"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestConfigAcceptsStrataEngineSelector(t *testing.T) {
	dir := t.TempDir()
	engine := filepath.Join(dir, "engine")
	script := "#!/bin/sh\nprintf 'READY 4096 stop\\n'\nwhile IFS= read -r line; do\n [ \"$line\" = QUIT ] && exit 0\ndone\n"
	if err := os.WriteFile(engine, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	config := standaloneConfig{Exe: engine, Tokenizer: lifecyclePack(t), Model: "test", Host: "127.0.0.1"}
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "run.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	invalid := exec.Command(exe, "-test.run=^TestLifecycleFixture$", "--", "server",
		"--engine", "other", "--config", path)
	invalid.Env = append(os.Environ(), "STRATA_LIFECYCLE_FIXTURE=1")
	output, err := invalid.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "--config accepts only --engine strata") {
		t.Fatalf("invalid engine selector: %v, %s", err, output)
	}
	cmd := exec.Command(exe, "-test.run=^TestLifecycleFixture$", "--", "server",
		"--engine", "strata", "--config", path, "--port", "0", "--api-key", "test-key")
	cmd.Env = append(os.Environ(), "STRATA_LIFECYCLE_FIXTURE=1")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waitStarted := false
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		if !waitStarted {
			_ = cmd.Wait()
		}
	})
	addresses := make(chan string, 1)
	go func() {
		defer close(addresses)
		scanner := bufio.NewScanner(stderr)
		sent := false
		for scanner.Scan() {
			if _, address, found := strings.Cut(scanner.Text(), "experimental Go server listening on "); found && !sent {
				addresses <- address
				sent = true
			}
		}
	}()
	var address string
	timer := time.NewTimer(20 * time.Second)
	defer timer.Stop()
	select {
	case value, ok := <-addresses:
		if !ok {
			t.Fatal("server exited before listening")
		}
		address = value
	case <-timer.C:
		t.Fatal("server did not start")
	}
	request, err := http.NewRequest(http.MethodGet, "http://"+address+"/v1/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer test-key")
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("models status %d", response.StatusCode)
	}
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	waitStarted = true
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-timer.C:
		t.Fatal("server did not exit")
	}
}
