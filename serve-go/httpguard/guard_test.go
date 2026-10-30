package httpguard

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func server(t *testing.T, c Config) *httptest.Server {
	t.Helper()
	h, err := New(c, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("handler read: %v", err)
		}
		_, _ = w.Write(b)
	}))
	if err != nil {
		t.Fatal(err)
	}
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	return s
}

func request(t *testing.T, s *httptest.Server, method, path, host, body string, headers map[string]string) *http.Response {
	t.Helper()
	r, err := http.NewRequest(method, s.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if host != "" {
		r.Host = host
	}
	for k, v := range headers {
		r.Header.Set(k, v)
	}
	res, err := s.Client().Do(r)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { res.Body.Close() })
	return res
}

func TestSecurityHTTP(t *testing.T) {
	s := server(t, Config{HostNames: []string{"box", ".example.org"}, CORSOrigins: []string{"https://chat.example.com"}, MaxBody: 8})
	cases := []struct {
		host, origin, content string
		status                int
	}{
		{"evil.example.com", "", "", 403},
		{"localhost.evil.com", "", "", 403},
		{"127.0.0.1.nip.io", "", "", 403},
		{"192.168.1.2:123", "", "", 200},
		{"[::1]:123", "", "", 200},
		{"app.localhost", "", "", 200},
		{"a.example.org", "", "", 200},
		{"localhost", "null", "application/json", 403},
		{"localhost", "http://192.168.1.2", "application/json", 403},
		{"localhost", "http://localhost:3000", "text/plain", 415},
		{"localhost", "http://localhost:3000", "application/json; charset=utf-8", 200},
		{"localhost", "chrome-extension://abc", "application/json", 200},
		{"localhost", "https://chat.example.com", "application/json", 200},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s_%s_%s", tc.host, tc.origin, tc.content), func(t *testing.T) {
			res := request(t, s, "POST", "/v1/chat", tc.host, "{}", map[string]string{"Origin": tc.origin, "Content-Type": tc.content})
			if res.StatusCode != tc.status {
				t.Fatalf("got %d want %d", res.StatusCode, tc.status)
			}
		})
	}
	r := request(t, s, "OPTIONS", "/v1/chat", "localhost", "", map[string]string{"Origin": "https://chat.example.com", "Access-Control-Request-Headers": "x-test"})
	if r.StatusCode != 204 || r.Header.Get("Access-Control-Allow-Origin") != "https://chat.example.com" || r.Header.Get("Access-Control-Allow-Headers") != "x-test" {
		t.Fatalf("preflight: %d %v", r.StatusCode, r.Header)
	}
	if got := request(t, s, "GET", "/health", "localhost", "", nil).StatusCode; got != 404 {
		t.Fatalf("unguarded route: %d", got)
	}
}

func TestKeysHTTP(t *testing.T) {
	s := server(t, Config{APIKeys: []string{"one", "café"}})
	for _, tc := range []struct {
		headers map[string]string
		status  int
	}{
		{nil, 401},
		{map[string]string{"X-Api-Key": "one"}, 200},
		{map[string]string{"Authorization": "bEaReR café"}, 200},
		{map[string]string{"Authorization": "Bearer wrong", "X-Api-Key": "one"}, 401},
		{map[string]string{"Authorization": "Basic x", "X-Api-Key": "one"}, 200},
	} {
		res := request(t, s, "POST", "/v1/chat", "tunnel.example", "{}", tc.headers)
		if res.StatusCode != tc.status {
			t.Fatalf("%v: got %d want %d", tc.headers, res.StatusCode, tc.status)
		}
	}
}

func TestPublicHealthIsLimitedToExactGetPaths(t *testing.T) {
	s := server(t, Config{APIKeys: []string{"secret"}, AllowHealth: true})
	for _, tc := range []struct {
		method, target string
		status         int
	}{
		{"GET", "/health", 200},
		{"GET", "/api/health?probe=1", 200},
		{"POST", "/health", 404},
		{"GET", "/health/", 404},
		{"GET", "/%68ealth", 404},
		{"GET", "/api/%68ealth", 404},
		{"GET", "/v1/models", 401},
	} {
		status, _ := raw(t, s, tc.method+" "+tc.target+" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
		if status != tc.status {
			t.Errorf("%s %s: got %d, want %d", tc.method, tc.target, status, tc.status)
		}
	}
	noKey := server(t, Config{AllowHealth: true})
	status, _ := raw(t, noKey, "GET /health HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n")
	if status != http.StatusForbidden {
		t.Fatalf("health with untrusted Host: got %d", status)
	}
}

func TestModelsAliasGuard(t *testing.T) {
	disabled := server(t, Config{APIKeys: []string{"secret"}})
	status, _ := raw(t, disabled, "GET /models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nConnection: close\r\n\r\n")
	if status != http.StatusNotFound {
		t.Fatalf("default guard forwarded alias: %d", status)
	}
	enabled := server(t, Config{APIKeys: []string{"secret"}, AllowModels: true})
	for _, tc := range []struct {
		target, auth string
		want         int
	}{
		{"/models", "", http.StatusUnauthorized},
		{"/models?x=1", "Authorization: Bearer secret\r\n", http.StatusOK},
		{"/%6dodels", "Authorization: Bearer secret\r\n", http.StatusNotFound},
		{"/models/", "Authorization: Bearer secret\r\n", http.StatusNotFound},
		{"/models/../settings", "Authorization: Bearer secret\r\n", http.StatusNotFound},
	} {
		status, _ := raw(t, enabled, "GET "+tc.target+" HTTP/1.1\r\nHost: localhost\r\n"+tc.auth+"Connection: close\r\n\r\n")
		if status != tc.want {
			t.Errorf("%s: got %d, want %d", tc.target, status, tc.want)
		}
	}
	noKey := server(t, Config{AllowModels: true})
	status, _ = raw(t, noKey, "GET /models HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n")
	if status != http.StatusForbidden {
		t.Fatalf("alias accepted untrusted Host: %d", status)
	}
}

func TestCanonicalPathBeforeDownstreamNormalization(t *testing.T) {
	var forwarded atomic.Int32
	h, err := New(Config{}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwarded.Add(1)
		if path.Clean(r.URL.Path) == "/settings" {
			w.WriteHeader(http.StatusTeapot)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	if err != nil {
		t.Fatal(err)
	}
	s := httptest.NewServer(h)
	defer s.Close()
	for _, target := range []string{
		"/v1/../settings", "/v1/./chat", "/v1/a/../../settings",
		"/v1/%2e%2e/settings", "/v1/%252e%252e/settings",
		"/v1/..%2fsettings", "/v1/%2e./settings", "/v1/%2f../settings",
		"/v1//chat", "/v1/..;/settings", "/v1/%5c..%5csettings",
		"/v1/chat%3fextra", "/v1/%63hat",
	} {
		t.Run(target, func(t *testing.T) {
			status, _ := raw(t, s, "GET "+target+" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
			if status != http.StatusNotFound {
				t.Fatalf("ambiguous path got status %d", status)
			}
		})
	}
	if got := forwarded.Load(); got != 0 {
		t.Fatalf("forwarded %d rejected paths", got)
	}
	for _, target := range []string{"/v1/chat", "/v1/chat/", "/v1/a-b_1.json?x=1"} {
		status, _ := raw(t, s, "GET "+target+" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
		if status != http.StatusNoContent {
			t.Fatalf("canonical path %q got status %d", target, status)
		}
	}
	if got := forwarded.Load(); got != 3 {
		t.Fatalf("forwarded %d canonical paths, want 3", got)
	}
}

func raw(t *testing.T, s *httptest.Server, wire string) (int, string) {
	t.Helper()
	conn, err := net.Dial("tcp", s.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(conn, wire); err != nil {
		t.Fatal(err)
	}
	if err := conn.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}
	resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, string(b)
}

func TestBodyFramingHTTP(t *testing.T) {
	s := server(t, Config{MaxBody: 4})
	head := "POST /v1/chat HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
	for _, tc := range []struct {
		name, wire string
		status     int
		body       string
	}{
		{"length", "Content-Length: 4\r\n\r\nabcd", 200, "abcd"},
		{"length-large", "Content-Length: 5\r\n\r\n", 413, ""},
		{"length-truncated", "Content-Length: 4\r\n\r\nab", 400, ""},
		{"chunked", "Transfer-Encoding: chunked\r\n\r\n2;ext=1\r\nab\r\n2\r\ncd\r\n0\r\nX-Test: yes\r\n\r\n", 200, "abcd"},
		{"chunked-large", "Transfer-Encoding: chunked\r\n\r\n5\r\nabcde\r\n0\r\n\r\n", 413, ""},
		{"chunked-invalid", "Transfer-Encoding: chunked\r\n\r\nzz\r\n", 400, ""},
		{"chunked-incomplete-trailer", "Transfer-Encoding: chunked\r\n\r\n2\r\nab\r\n0\r\nX-Test: x\r\n", 400, ""},
		{"chunked-over-length", "Transfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n2\r\nab\r\n0\r\n\r\n", 200, "ab"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, body := raw(t, s, head+tc.wire)
			if status != tc.status || tc.status == 200 && body != tc.body {
				t.Fatalf("got %d %q, want %d %q", status, body, tc.status, tc.body)
			}
		})
	}
}

func TestBadConfig(t *testing.T) {
	for _, c := range []Config{
		{MaxBody: -1}, {APIKeys: []string{""}}, {HostNames: []string{"bad host"}},
		{TrustedOrigins: []string{"*"}}, {CORSOrigins: []string{"http://bad/path"}},
	} {
		if _, err := New(c, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})); err == nil {
			t.Fatalf("accepted %+v", c)
		}
	}
}
