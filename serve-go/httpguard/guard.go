// Package httpguard provides an API-only HTTP boundary, not a complete Strata server.
package httpguard

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"path"
	"strings"
	"unicode"
)

const DefaultMaxBody int64 = 256 << 20

// Config is copied by New. Names are normalized host names, optionally starting
// with a dot for a domain and its subdomains. Origins are exact strings.
type Config struct {
	APIKeys        []string
	HostNames      []string
	AllowAnyHost   bool
	AllowHealth    bool
	AllowModels    bool
	TrustedOrigins []string
	CORSOrigins    []string
	MaxBody        int64
}

// New wraps /v1/ routes and, when enabled, health and the /models alias.
// Other paths require separate route-specific guards.
func New(c Config, next http.Handler) (http.Handler, error) {
	if next == nil || c.MaxBody < 0 {
		return nil, errors.New("httpguard: invalid handler or body limit")
	}
	if c.MaxBody == 0 {
		c.MaxBody = DefaultMaxBody
	}
	c.APIKeys = append([]string(nil), c.APIKeys...)
	c.HostNames = append([]string(nil), c.HostNames...)
	c.TrustedOrigins = append([]string(nil), c.TrustedOrigins...)
	c.CORSOrigins = append([]string(nil), c.CORSOrigins...)
	for _, k := range c.APIKeys {
		if k == "" || strings.TrimSpace(k) != k {
			return nil, errors.New("httpguard: API keys must be nonempty and trimmed")
		}
	}
	for _, n := range c.HostNames {
		if n == "" || n != strings.ToLower(n) || hostName(strings.TrimPrefix(n, ".")) != strings.TrimPrefix(n, ".") {
			return nil, errors.New("httpguard: invalid normalized host name")
		}
	}
	for _, o := range append(append([]string{}, c.TrustedOrigins...), c.CORSOrigins...) {
		if o == "*" {
			if contains(c.TrustedOrigins, "*") {
				return nil, errors.New("httpguard: trusted origin wildcard is unsupported")
			}
			continue
		}
		if !(strings.HasPrefix(o, "http://") || strings.HasPrefix(o, "https://")) ||
			strings.Contains(strings.SplitN(o, "://", 2)[1], "/") ||
			hostName(strings.SplitN(o, "://", 2)[1]) == "" {
			return nil, errors.New("httpguard: invalid origin")
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(c.APIKeys) == 0 && !hostAllowed(r.Host, c.HostNames, c.AllowAnyHost) {
			fail(w, 403, "forbidden", "Host is not allowed")
			return
		}
		health := c.AllowHealth && r.Method == http.MethodGet &&
			(r.URL.Path == "/health" || r.URL.Path == "/api/health") &&
			r.URL.RawPath == "" && r.URL.EscapedPath() == r.URL.Path
		models := c.AllowModels && r.URL.Path == "/models" &&
			r.URL.RawPath == "" && r.URL.EscapedPath() == r.URL.Path
		if !health && !models && !canonicalAPIPath(r) {
			fail(w, 404, "not_found", "route is outside the API guard")
			return
		}
		if !health {
			cors(w, r, c.CORSOrigins)
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if !health && len(c.APIKeys) > 0 && !authorized(r, c.APIKeys) {
			fail(w, 401, "authentication_error", "missing or wrong API key")
			return
		}
		if r.Method == http.MethodPost && len(c.APIKeys) == 0 && r.Header.Get("Origin") != "" {
			if !originAllowed(r.Header.Get("Origin"), r.Host, c) {
				fail(w, 403, "forbidden", "origin is not allowed")
				return
			}
			if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
				fail(w, 415, "", "send application/json")
				return
			}
		}
		if r.ContentLength > c.MaxBody {
			fail(w, 413, "", "request body too large")
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, c.MaxBody))
		if err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				fail(w, 413, "", "request body too large")
			} else {
				fail(w, 400, "", "malformed or incomplete request body")
			}
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		next.ServeHTTP(w, r)
	}), nil
}

// Reject escaped bytes and separators that another router might reinterpret.
// The restricted alphabet also excludes semicolon path parameters and controls.
func canonicalAPIPath(r *http.Request) bool {
	p := r.URL.Path
	if !strings.HasPrefix(p, "/v1/") || r.URL.RawPath != "" ||
		strings.Contains(r.URL.EscapedPath(), "%") || strings.Contains(p, "//") {
		return false
	}
	for _, c := range p {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' ||
			c >= '0' && c <= '9' || strings.ContainsRune("/-._~", c)) {
			return false
		}
	}
	clean := path.Clean(p)
	return p == clean || strings.HasSuffix(p, "/") && strings.TrimSuffix(p, "/") == clean
}

func fail(w http.ResponseWriter, code int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	e := map[string]string{"message": msg}
	if kind != "" {
		e["type"] = kind
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"error": e})
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

func hostName(value string) string {
	v := strings.ToLower(strings.TrimSpace(value))
	if strings.HasPrefix(v, "[") {
		i := strings.IndexByte(v, ']')
		if i < 0 || (len(v) > i+1 && (v[i+1] != ':' || !digits(v[i+2:]))) {
			return ""
		}
		return v[1:i]
	}
	if strings.Count(v, ":") == 1 {
		var port string
		v, port, _ = strings.Cut(v, ":")
		if !digits(port) {
			return ""
		}
	} else if strings.Contains(v, ":") {
		return v
	}
	v = strings.TrimRight(v, ".")
	for _, r := range v {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && !strings.ContainsRune("-._", r) {
			return ""
		}
	}
	return v
}

func digits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func named(n string, names []string) bool {
	if n == "localhost" || n == "127.0.0.1" || n == "::1" || strings.HasSuffix(n, ".localhost") {
		return true
	}
	for _, x := range names {
		if n == x || strings.HasPrefix(x, ".") && (n == x[1:] || strings.HasSuffix(n, x)) {
			return true
		}
	}
	return false
}

func hostAllowed(host string, names []string, any bool) bool {
	if any || strings.TrimSpace(host) == "" {
		return true
	}
	n := hostName(host)
	return n != "" && (net.ParseIP(n) != nil || named(n, names))
}

func originAllowed(origin, host string, c Config) bool {
	origin = strings.TrimRight(strings.TrimSpace(origin), "/")
	if contains(c.TrustedOrigins, origin) || contains(c.CORSOrigins, origin) || contains(c.CORSOrigins, "*") {
		return true
	}
	scheme, rest, ok := strings.Cut(strings.ToLower(origin), "://")
	if !ok || scheme == "" {
		return false
	}
	if scheme != "http" && scheme != "https" {
		return true
	}
	if host != "" && rest == strings.ToLower(strings.TrimSpace(host)) {
		return true
	}
	n := hostName(rest)
	return n != "" && named(n, c.HostNames)
}

func authorized(r *http.Request, keys []string) bool {
	auth := r.Header.Get("Authorization")
	given := r.Header.Get("X-Api-Key")
	if len(auth) >= 7 && strings.EqualFold(auth[:7], "bearer ") {
		given = strings.TrimSpace(auth[7:])
	}
	matched := 0
	for _, key := range keys {
		matched |= subtle.ConstantTimeCompare([]byte(given), []byte(key))
		latin := make([]byte, 0, len(key))
		valid := true
		for _, ch := range key {
			if ch > 255 {
				valid = false
				break
			}
			latin = append(latin, byte(ch))
		}
		if valid {
			matched |= subtle.ConstantTimeCompare([]byte(given), latin)
		}
	}
	return matched != 0
}

func cors(w http.ResponseWriter, r *http.Request, origins []string) {
	origin := strings.TrimRight(r.Header.Get("Origin"), "/")
	allow := ""
	if contains(origins, "*") {
		allow = "*"
	} else if origin != "" && contains(origins, origin) {
		allow = origin
		w.Header().Add("Vary", "Origin")
	}
	if allow == "" {
		return
	}
	w.Header().Set("Access-Control-Allow-Origin", allow)
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	asked := r.Header.Get("Access-Control-Request-Headers")
	if asked == "" {
		asked = "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta"
	}
	w.Header().Set("Access-Control-Allow-Headers", asked)
	w.Header().Set("Access-Control-Max-Age", "600")
}
