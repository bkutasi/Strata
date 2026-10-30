// Package visioninput loads image request sources and assembles image files for GENI.
package visioninput

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const maxImageSize = 32 << 20

// Load reads a base64 data URL, HTTP(S) URL, or local image file.
func Load(ctx context.Context, source string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var r io.Reader
	var closeFn func() error
	switch {
	case strings.HasPrefix(source, "data:"):
		meta, data, ok := strings.Cut(source[len("data:"):], ",")
		if !ok || !strings.HasSuffix(strings.ToLower(meta), ";base64") {
			return nil, fmt.Errorf("image data URL must contain base64 data")
		}
		r = base64.NewDecoder(base64.StdEncoding, strings.NewReader(data))
	case strings.HasPrefix(source, "http://"), strings.HasPrefix(source, "https://"):
		requestCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, source, nil)
		if err != nil {
			return nil, fmt.Errorf("invalid image URL: %w", err)
		}
		req.Header.Set("User-Agent", "strata")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("image URL could not be read: %w", err)
		}
		if resp.ContentLength > maxImageSize {
			resp.Body.Close()
			return nil, fmt.Errorf("image URL file is over 32 MiB")
		}
		r, closeFn = resp.Body, resp.Body.Close
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			closeFn()
			return nil, fmt.Errorf("image URL returned HTTP %s", resp.Status)
		}
	case strings.HasPrefix(source, "file://"):
		path, err := localFileURL(source)
		if err != nil {
			return nil, err
		}
		f, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("image file %q could not be read: %w", path, err)
		}
		r, closeFn = f, f.Close
	default:
		if networkPath(source) {
			return nil, fmt.Errorf("image file must be on this computer: network paths are not read")
		}
		f, err := os.Open(source)
		if err != nil {
			return nil, fmt.Errorf("image file %q could not be read: %w", source, err)
		}
		r, closeFn = f, f.Close
	}
	if closeFn != nil {
		defer closeFn()
	}
	data, err := io.ReadAll(io.LimitReader(r, maxImageSize+1))
	if err != nil {
		return nil, fmt.Errorf("image could not be read: %w", err)
	}
	if len(data) > maxImageSize {
		return nil, fmt.Errorf("image file is over 32 MiB")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return data, nil
}

func localFileURL(source string) (string, error) {
	raw := strings.TrimPrefix(source, "file://")
	if networkPath(raw) {
		return "", fmt.Errorf("image file must be on this computer: network paths are not read")
	}
	host, path, hasPath := strings.Cut(raw, "/")
	if strings.Contains(raw, `\`) {
		host, path, hasPath = strings.Cut(raw, `\`)
		path = strings.ReplaceAll(path, `\`, string(os.PathSeparator))
	}
	if host != "" && !strings.EqualFold(host, "localhost") && !isDrive(host) {
		return "", fmt.Errorf("image file must be on this computer: network paths are not read")
	}
	if strings.EqualFold(host, "localhost") {
		if !hasPath {
			return "", fmt.Errorf("file URL has no local path")
		}
		return string(os.PathSeparator) + path, nil
	}
	if isDrive(host) {
		if !hasPath {
			return host, nil
		}
		return host + string(os.PathSeparator) + path, nil
	}
	if host == "" {
		return string(os.PathSeparator) + path, nil
	}
	return "", fmt.Errorf("invalid local file URL")
}

func isDrive(host string) bool {
	return len(host) == 2 && ((host[0] >= 'A' && host[0] <= 'Z') || (host[0] >= 'a' && host[0] <= 'z')) && host[1] == ':'
}

func networkPath(path string) bool {
	return strings.HasPrefix(path, `\\`) || strings.HasPrefix(path, "//")
}

// Combine concatenates files in order into a private temporary file for GENI.
// The caller owns and must remove the returned file.
func Combine(paths []string) (string, error) {
	out, err := os.CreateTemp("", "strata-visioninput-*.sve")
	if err != nil {
		return "", fmt.Errorf("create image assembly file: %w", err)
	}
	outPath := out.Name()
	fail := func(err error) (string, error) {
		_ = out.Close()
		_ = os.Remove(outPath)
		return "", err
	}
	if strings.ContainsAny(outPath, " \t\r\n\v\f=\x00") {
		return fail(fmt.Errorf("temporary path contains characters unsafe for GENI: %q", outPath))
	}
	for _, path := range paths {
		in, err := os.Open(path)
		if err != nil {
			return fail(fmt.Errorf("open image file %q: %w", path, err))
		}
		_, copyErr := io.Copy(out, in)
		closeErr := in.Close()
		if copyErr != nil {
			return fail(fmt.Errorf("append image file %q: %w", path, copyErr))
		}
		if closeErr != nil {
			return fail(fmt.Errorf("close image file %q: %w", path, closeErr))
		}
	}
	if err := out.Close(); err != nil {
		return fail(fmt.Errorf("close image assembly file: %w", err))
	}
	return outPath, nil
}
