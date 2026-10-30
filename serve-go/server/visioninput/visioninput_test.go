package visioninput

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadSources(t *testing.T) {
	dir := t.TempDir()
	local := filepath.Join(dir, "image.bin")
	if err := os.WriteFile(local, []byte("local image"), 0600); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/large" {
			w.Header().Set("Content-Length", "33554433")
			return
		}
		_, _ = w.Write([]byte("remote image"))
	}))
	defer server.Close()
	cases := []struct {
		name, source, want string
	}{
		{"data URL", "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("data image")), "data image"},
		{"HTTP URL", server.URL, "remote image"},
		{"local path", local, "local image"},
		{"file URL", "file://" + local, "local image"},
		{"localhost file URL", "file://localhost" + local, "local image"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Load(context.Background(), tc.source)
			if err != nil || string(got) != tc.want {
				t.Fatalf("Load() = %q, %v; want %q", got, err, tc.want)
			}
		})
	}
	for _, source := range []string{"file://server/share/image.png", `\\server\share\image.png`, "//server/share/image.png"} {
		if _, err := Load(context.Background(), source); err == nil {
			t.Errorf("Load(%q) unexpectedly succeeded", source)
		}
	}
	if _, err := Load(context.Background(), "data:image/png;base64,@@"); err == nil {
		t.Error("invalid base64 unexpectedly succeeded")
	}
	if _, err := Load(context.Background(), server.URL+"/large"); err == nil {
		t.Error("oversized HTTP image unexpectedly succeeded")
	}
}

func TestLoadSizeLimit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "large")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxImageSize + 1); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(context.Background(), path); err == nil {
		t.Fatal("oversized local image unexpectedly succeeded")
	}
}

func TestCombine(t *testing.T) {
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "one"), filepath.Join(dir, "two")}
	for i, content := range []string{"embedding-one", "embedding-two"} {
		if err := os.WriteFile(paths[i], []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	combined, err := Combine(paths)
	if err != nil {
		t.Fatal(err)
	}
	if strings.ContainsAny(combined, " \t\r\n=") {
		t.Fatalf("unsafe GENI path %q", combined)
	}
	info, err := os.Stat(combined)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0600 {
		t.Fatalf("permissions = %o, want 600", info.Mode().Perm())
	}
	data, err := os.ReadFile(combined)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "embedding-oneembedding-two" {
		t.Fatalf("combined data = %q", data)
	}
	if err := os.Remove(combined); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(combined); !os.IsNotExist(err) {
		t.Fatalf("caller removal failed: %v", err)
	}
}

func TestCombineCleansUpOnError(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "first")
	if err := os.WriteFile(first, []byte("partial"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Combine([]string{first, filepath.Join(dir, "missing")}); err == nil {
		t.Fatal("Combine with missing input unexpectedly succeeded")
	}
}
