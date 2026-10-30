// strata-go-server is an experimental opt-in text-only server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"strata/engine"
	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
	"strata/server"
)

type args []string

func (a *args) String() string     { return fmt.Sprint([]string(*a)) }
func (a *args) Set(s string) error { *a = append(*a, s); return nil }

func main() {
	var engineArgs args
	listen := flag.String("listen", "127.0.0.1:8080", "listen address")
	model := flag.String("model", "", "model name exposed to clients")
	pack := flag.String("pack", "", "model pack or tokenizer directory")
	engineExe := flag.String("engine", "", "native executable")
	keyEnv := flag.String("api-key-env", "", "environment variable containing API key")
	flag.Var(&engineArgs, "engine-arg", "native engine argument (repeat; include --serve and model flags)")
	flag.Parse()
	if flag.NArg() != 0 || *model == "" || *pack == "" || *engineExe == "" {
		flag.Usage()
		os.Exit(2)
	}
	host, _, err := net.SplitHostPort(*listen)
	if err != nil {
		log.Fatal(err)
	}
	var keys []string
	if *keyEnv != "" {
		key := os.Getenv(*keyEnv)
		if key == "" {
			log.Fatal("API key environment variable is empty")
		}
		keys = []string{key}
	}
	ip := net.ParseIP(host)
	if len(keys) == 0 && host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		log.Fatal("non-loopback listening requires an API key")
	}
	tok, err := modeltokenizer.Load(*pack)
	if err != nil {
		log.Fatal(err)
	}
	renderer, err := server.NewTemplateRenderer(*pack)
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	eng, err := engine.Start(ctx, *engineExe, engineArgs, os.Stderr)
	cancel()
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := eng.Close(ctx); err != nil {
			log.Printf("engine close: %v", err)
		}
	}()
	handler, err := server.New(server.Config{Model: *model, Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: eng, Renderer: renderer, APIKeys: keys, MaxBody: 2 << 20, MaxPromptBytes: 2 << 20, MaxOutputTokens: 4096, MaxConcurrent: 4})
	if err != nil {
		log.Fatal(err)
	}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatal(err)
	}
	httpServer := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 1 << 20}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdown); err != nil {
			log.Printf("shutdown: %v", err)
		}
	}()
	log.Printf("experimental Go server listening on %s", ln.Addr())
	if err = httpServer.Serve(ln); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
