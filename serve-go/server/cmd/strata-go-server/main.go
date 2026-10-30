// strata-go-server is an experimental opt-in text-only server.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
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
	"strata/server/vision"
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
	configPath := flag.String("config", "", "standalone Strata JSON config")
	port := flag.Int("port", 8080, "API port when using --config")
	apiKey := flag.String("api-key", "", "API key (prefer --api-key-env)")
	noResponses := flag.Bool("no-responses", false, "disable the /v1/responses endpoint")
	noAnthropic := flag.Bool("no-anthropic", false, "disable Anthropic Messages and count_tokens endpoints")
	lazyVision := flag.Bool("lazy-vision", false, "start the vision reader only for image requests and stop it afterwards")
	flag.Var(&engineArgs, "engine-arg", "native engine argument (repeat; include --serve and model flags)")
	flag.Parse()
	var processConfig engine.ProcessConfig
	var engineLog io.Writer = os.Stderr
	var configuredKey json.RawMessage
	var defaultReasoningBudget *int
	var repeatStopTokens *int
	var defaultSampling engine.Sampling
	var defaultReasoningEffort string
	var aliases []string
	var anthropicThinkingOnRequest bool
	var disableAnswerRoom bool
	var disableFitMaxTokens bool
	var standaloneVision *standaloneVisionConfig
	var visionArgs []string
	var standalone *standaloneConfig
	if *configPath != "" {
		cfg, err := readStandaloneConfig(*configPath)
		if err != nil {
			return err
		}
		standalone = &cfg
		defaultReasoningBudget, disableAnswerRoom, err = cfg.thinkingConfig()
		if err != nil {
			return err
		}
		disableFitMaxTokens = bytes.Equal(bytes.TrimSpace(cfg.FitMaxTokens), []byte("false"))
		repeatStopTokens, err = cfg.repeatStopConfig()
		if err != nil {
			return err
		}
		anthropicThinkingOnRequest = cfg.AnthropicThinking == "on_request"
		defaultReasoningEffort = cfg.ReasoningEffort
		aliases, err = cfg.aliasesConfig()
		if err != nil {
			return err
		}
		if len(cfg.Sampling) > 0 {
			defaultSampling, err = server.ParseSamplingDefaults(cfg.Sampling)
			if err != nil {
				return fmt.Errorf("invalid config sampling: %w", err)
			}
		}
		set := make(map[string]bool)
		flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
		if (set["engine"] && *engineExe != "strata") || set["engine-arg"] || set["pack"] || set["model"] {
			return errors.New("--config accepts only --engine strata and cannot be combined with engine-arg, pack or model")
		}
		*engineExe, *pack, *model = cfg.Exe, cfg.Tokenizer, cfg.Model
		engineArgs = append(args{"--serve"}, cfg.Args...)
		processConfig.Dir = cfg.Cwd
		processConfig.Env, err = cfg.childEnv()
		if err != nil {
			return err
		}
		if !set["listen"] {
			host := cfg.Host
			if host == "" {
				host = "127.0.0.1"
			}
			*listen = net.JoinHostPort(host, strconv.Itoa(*port))
		} else if set["port"] {
			return errors.New("--port cannot be combined with --listen")
		}
		configuredKey = cfg.APIKey
		if cfg.Log != "" {
			logFile, err := os.OpenFile(cfg.Log, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
			if err != nil {
				return err
			}
			defer func() {
				if err := logFile.Close(); err != nil {
					log.Printf("engine log close: %v", err)
				}
			}()
			engineLog = logFile
		}
	}
	if flag.NArg() != 0 || *model == "" || *pack == "" || *engineExe == "" {
		flag.Usage()
		os.Exit(2)
	}
	selectedFamily := requestnorm.Family(*family)
	if selectedFamily != requestnorm.FamilyQwen && selectedFamily != requestnorm.FamilyDeepSeek {
		return fmt.Errorf("unsupported model family %q", *family)
	}
	var err error
	if *configPath != "" {
		standaloneVision, visionArgs, err = standalone.visionConfig(selectedFamily)
		if err != nil {
			return fmt.Errorf("invalid config: %w", err)
		}
		if standaloneVision != nil {
			hasVision, hasReserve := false, false
			for _, arg := range engineArgs {
				if arg == "--vision" {
					hasVision = true
				}
				if arg == "--vram-reserve-mib" {
					hasReserve = true
				}
			}
			if !hasVision {
				engineArgs = append(engineArgs, "--vision")
				if standaloneVision.GPU && !hasReserve {
					engineArgs = append(engineArgs, "--vram-reserve-mib", "700")
				}
			}
		}
	}
	host, _, err := net.SplitHostPort(*listen)
	if err != nil {
		return err
	}
	apiKeySet := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "api-key" {
			apiKeySet = true
		}
	})
	var keys []string
	if *keyEnv != "" {
		key := os.Getenv(*keyEnv)
		if key == "" {
			return fmt.Errorf("API key environment variable is empty")
		}
		keys = []string{key}
	} else if apiKeySet {
		if *apiKey == "" {
			return errors.New("API key is empty")
		}
		keys, err = splitAPIKeys(*apiKey)
	} else if key, ok := os.LookupEnv("STRATA_API_KEY"); ok {
		if key == "" {
			return errors.New("API key is empty")
		}
		keys, err = splitAPIKeys(key)
	} else {
		keys, err = configAPIKeys(configuredKey)
	}
	if err != nil {
		return err
	}
	ip := net.ParseIP(host)
	if len(keys) == 0 && host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("non-loopback listening requires an API key")
	}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return err
	}
	defer ln.Close()
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
		return engine.StartWithProcessConfig(ctx, *engineExe, engineArgs, engineLog, processConfig)
	}
	if ctx.Err() != nil {
		return nil
	}
	var visionReader interface {
		Encode(context.Context, []byte) (string, int, error)
		Close() error
	}
	if standaloneVision != nil {
		if *lazyVision {
			visionReader = vision.NewLazy(visionArgs[0], visionArgs[1:], processConfig.Env)
		} else {
			visionCtx, visionCancel := context.WithTimeout(ctx, 300*time.Second)
			reader, startErr := vision.StartWithEnv(visionCtx, visionArgs[0], visionArgs[1:], processConfig.Env)
			visionCancel()
			if startErr != nil {
				return startErr
			}
			visionReader = reader
		}
		defer func() {
			if err := visionReader.Close(); err != nil {
				log.Printf("vision close: %v", err)
			}
		}()
	}
	logOffset := int64(0)
	logPath := ""
	if f, ok := engineLog.(*os.File); ok {
		if st, statErr := f.Stat(); statErr != nil {
			log.Printf("startup narration log: %v", statErr)
		} else if st.Mode().IsRegular() {
			logOffset = st.Size()
			logPath = f.Name()
		}
	}
	narrationDone := make(chan struct{})
	narrationJoined := make(chan struct{})
	go narrateStart(logPath, logOffset, narrationDone, narrationJoined, 500*time.Millisecond, time.Now, func(e narrationEvent) {
		switch e.kind {
		case "weights":
			fmt.Fprintln(os.Stderr, "[strata] starting the engine: reading the model's weights ...")
		case "heartbeat":
			fmt.Fprintf(os.Stderr, "[strata] still starting (%s s) - please wait ...\n", e.data)
		case "arena":
			fmt.Fprintln(os.Stderr, "[strata] loading the experts\n         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n         Please wait and don't close this window; the server says when it is ready.")
		case "loaded":
			fmt.Fprintf(os.Stderr, "[strata] experts loaded: %s\n", e.data)
		case "cache":
			fmt.Fprintf(os.Stderr, "[strata] filling the GPU's expert cache (%s) ...\n", e.data)
		case "up":
			fmt.Fprintln(os.Stderr, "[strata] almost ready ...")
		case "log_error":
			log.Printf("startup narration log: %s", e.data)
		}
	})
	eng, err := start(ctx)
	close(narrationDone)
	<-narrationJoined
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
	maxBody := int64(2 << 20)
	if visionReader != nil {
		maxBody = 48 << 20
	}
	serverConfig := server.Config{Model: *model, Aliases: aliases, Family: selectedFamily, Tokenizer: tok, Engine: supervisor, Renderer: renderer, APIKeys: keys, MaxBody: maxBody, MaxPromptBytes: 2 << 20, MaxOutputTokens: int(^uint(0) >> 1), MaxConcurrent: 4, DefaultReasoningBudget: defaultReasoningBudget, DisableAnswerRoom: disableAnswerRoom, DisableFitMaxTokens: disableFitMaxTokens, DisableResponses: *noResponses, DisableAnthropic: *noAnthropic, RepeatStopTokens: repeatStopTokens, AnthropicThinkingOnRequest: anthropicThinkingOnRequest, DefaultSampling: defaultSampling, DefaultReasoningEffort: defaultReasoningEffort, StatusLogger: log.New(os.Stderr, "", 0)}
	if visionReader != nil {
		serverConfig.Vision = visionReader
	}
	handler, err := server.New(serverConfig)
	if err != nil {
		return err
	}
	// Release temporary vocabulary and template-loading buffers before serving.
	debug.FreeOSMemory()
	httpServer := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 1 << 20}
	log.Printf("experimental Go server listening on %s", ln.Addr())
	return serveUntilShutdown(ctx, httpServer, ln)
}
