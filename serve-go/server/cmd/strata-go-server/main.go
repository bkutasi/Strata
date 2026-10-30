// strata-go-server is an experimental opt-in text-only server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"strconv"
	"syscall"
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
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	var engineArgs args
	listen := flag.String("listen", "127.0.0.1:8080", "listen address")
	model := flag.String("model", "", "model name exposed to clients")
	family := flag.String("family", "qwen", "model family (qwen or deepseek41)")
	pack := flag.String("pack", "", "model pack or tokenizer directory")
	engineExe := flag.String("engine", "", "native executable")
	keyEnv := flag.String("api-key-env", "", "environment variable containing API key")
	ownerFD := flag.String("owner-fd", "", "inherited owner pipe descriptor")
	flag.Var(&engineArgs, "engine-arg", "native engine argument (repeat; include --serve and model flags)")
	flag.Parse()
	if flag.NArg() != 0 || *model == "" || *pack == "" || *engineExe == "" {
		flag.Usage()
		os.Exit(2)
	}
	selectedFamily := requestnorm.Family(*family)
	if selectedFamily != requestnorm.FamilyQwen && selectedFamily != requestnorm.FamilyDeepSeek {
		return fmt.Errorf("unsupported model family %q", *family)
	}
	host, _, err := net.SplitHostPort(*listen)
	if err != nil {
		return err
	}
	var keys []string
	if *keyEnv != "" {
		key := os.Getenv(*keyEnv)
		if key == "" {
			return fmt.Errorf("API key environment variable is empty")
		}
		keys = []string{key}
	}
	ip := net.ParseIP(host)
	if len(keys) == 0 && host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("non-loopback listening requires an API key")
	}
	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	ctx, cancel := context.WithCancel(signalCtx)
	var owner *os.File
	defer func() {
		cancel()
		if owner != nil {
			if err := owner.Close(); err != nil {
				log.Printf("owner pipe close: %v", err)
			}
		}
		stop()
	}()
	if *ownerFD != "" {
		fd, err := strconv.Atoi(*ownerFD)
		if err != nil {
			return fmt.Errorf("--owner-fd must be an inherited pipe")
		}
		owner, err = startOwnerGuard(ctx, cancel, fd)
		if err != nil {
			return err
		}
	}
	tok, err := modeltokenizer.Load(*pack)
	if err != nil {
		return err
	}
	renderer, err := server.NewTemplateRenderer(*pack, tok)
	if err != nil {
		return err
	}
	start := func(ctx context.Context) (*engine.Engine, error) {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		return engine.Start(ctx, *engineExe, engineArgs, os.Stderr)
	}
	if ctx.Err() != nil {
		return nil
	}
	eng, err := start(ctx)
	if err != nil {
		if errors.Is(err, context.Canceled) && !errors.Is(err, engine.ErrUnreaped) {
			return nil
		}
		return err
	}
	supervisor, err := engine.Supervise(eng, start)
	if err != nil {
		closeCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		return errors.Join(err, eng.Close(closeCtx))
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := supervisor.Close(ctx); err != nil {
			log.Printf("engine close: %v", err)
		}
	}()
	handler, err := server.New(server.Config{Model: *model, Family: selectedFamily, Tokenizer: tok, Engine: supervisor, Renderer: renderer, APIKeys: keys, MaxBody: 2 << 20, MaxPromptBytes: 2 << 20, MaxOutputTokens: int(^uint(0) >> 1), MaxConcurrent: 4})
	if err != nil {
		return err
	}
	// Release temporary vocabulary and template-loading buffers before serving.
	debug.FreeOSMemory()
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return err
	}
	httpServer := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 1 << 20}
	log.Printf("experimental Go server listening on %s", ln.Addr())
	return serveUntilShutdown(ctx, httpServer, ln)
}
