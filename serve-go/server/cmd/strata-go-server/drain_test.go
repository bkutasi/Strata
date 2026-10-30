package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"testing"
	"time"
)

func TestServerDrainsActiveRequestOnShutdown(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	entered := make(chan struct{})
	release := make(chan struct{})
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		select {
		case <-release:
			_, _ = io.WriteString(w, "done")
		case <-r.Context().Done():
		}
	})}
	shutdown := make(chan struct{})
	server.RegisterOnShutdown(func() { close(shutdown) })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- serveUntilShutdown(ctx, server, ln) }()
	requestDone := make(chan error, 1)
	go func() {
		response, err := (&http.Client{Timeout: 10 * time.Second}).Get("http://" + ln.Addr().String())
		if err == nil {
			var body []byte
			body, err = io.ReadAll(response.Body)
			_ = response.Body.Close()
			if err == nil && string(body) != "done" {
				err = io.ErrUnexpectedEOF
			}
		}
		requestDone <- err
	}()
	timer := time.NewTimer(10 * time.Second)
	defer timer.Stop()
	select {
	case <-entered:
	case <-timer.C:
		t.Fatal("request did not enter handler")
	}
	cancel()
	select {
	case <-shutdown:
	case <-timer.C:
		t.Fatal("shutdown did not start")
	}
	select {
	case err := <-done:
		t.Fatalf("server returned before draining: %v", err)
	default:
	}
	close(release)
	select {
	case err := <-requestDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-timer.C:
		t.Fatal("request did not finish")
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-timer.C:
		t.Fatal("server did not finish shutdown")
	}
}
