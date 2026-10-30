// strata-go-server is an experimental opt-in text-only server.
package main

import (
	"bytes"
	"context"
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
	"strings"
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
	flag.Var(&engineArgs, "engine-arg", "native engine argument (repeat; include --serve and model flags)")
	flag.Parse()
	var processConfig engine.ProcessConfig
	var engineLog io.Writer = os.Stderr
	var configuredKey string
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
	var keys []string
	if *apiKey != "" {
		configuredKey = *apiKey
	} else if key := os.Getenv("STRATA_API_KEY"); key != "" {
		configuredKey = key
	}
	if *keyEnv != "" {
		key := os.Getenv(*keyEnv)
		if key == "" {
			return fmt.Errorf("API key environment variable is empty")
		}
		keys = []string{key}
	} else if configuredKey != "" && !strings.EqualFold(strings.TrimSpace(configuredKey), "none") {
		keys = []string{configuredKey}
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
		return engine.StartWithProcessConfig(ctx, *engineExe, engineArgs, engineLog, processConfig)
	}
	if ctx.Err() != nil {
		return nil
	}
	var visionReader *vision.Reader
	if standaloneVision != nil {
		visionCtx, visionCancel := context.WithTimeout(ctx, 300*time.Second)
		visionReader, err = vision.StartWithEnv(visionCtx, visionArgs[0], visionArgs[1:], processConfig.Env)
		visionCancel()
		if err != nil {
			return err
		}
		defer func() {
			if err := visionReader.Close(); err != nil {
				log.Printf("vision close: %v", err)
			}
		}()
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
	maxBody := int64(2 << 20)
	if visionReader != nil {
		maxBody = 48 << 20
	}
	serverConfig := server.Config{Model: *model, Aliases: aliases, Family: selectedFamily, Tokenizer: tok, Engine: supervisor, Renderer: renderer, APIKeys: keys, MaxBody: maxBody, MaxPromptBytes: 2 << 20, MaxOutputTokens: int(^uint(0) >> 1), MaxConcurrent: 4, DefaultReasoningBudget: defaultReasoningBudget, DisableAnswerRoom: disableAnswerRoom, DisableFitMaxTokens: disableFitMaxTokens, RepeatStopTokens: repeatStopTokens, AnthropicThinkingOnRequest: anthropicThinkingOnRequest, DefaultSampling: defaultSampling, DefaultReasoningEffort: defaultReasoningEffort}
	if visionReader != nil {
		serverConfig.Vision = visionReader
	}
	handler, err := server.New(serverConfig)
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
