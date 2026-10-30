package server

import (
	"io"
	"net/http"
)

// streamHeartbeat forwards native prompt progress as an SSE comment.
func streamHeartbeat(w http.ResponseWriter, stream bool, started *bool) func() error {
	if !stream {
		return nil
	}
	return func() error {
		if !*started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			*started = true
		}
		if _, err := io.WriteString(w, ": keep-alive\n\n"); err != nil {
			return err
		}
		return http.NewResponseController(w).Flush()
	}
}
