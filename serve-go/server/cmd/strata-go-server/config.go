package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"strata/requestnorm"
)

var joinedConfigArg = regexp.MustCompile(`^(--[A-Za-z0-9][A-Za-z0-9-]*)([[:space:]]+|=)([^[:space:]].*)$`)
var bareConfigFlag = regexp.MustCompile(`^--[A-Za-z0-9][A-Za-z0-9-]*$`)

// standaloneConfig supports the fields in the standalone Strata run config.
// Unknown settings are rejected rather than silently ignored.
type standaloneConfig struct {
	Exe               string            `json:"exe"`
	Cwd               string            `json:"cwd"`
	Args              []string          `json:"args"`
	Tokenizer         string            `json:"tokenizer"`
	Model             string            `json:"model_name"`
	Aliases           json.RawMessage   `json:"aliases,omitempty"`
	Host              string            `json:"host"`
	APIKey            json.RawMessage   `json:"api_key"`
	GPU               *int              `json:"gpu"`
	Env               map[string]string `json:"env"`
	Log               string            `json:"log"`
	AnswerRoom        json.RawMessage   `json:"answer_room"`
	FitMaxTokens      json.RawMessage   `json:"fit_max_tokens,omitempty"`
	ReasoningBudget   json.RawMessage   `json:"reasoning_budget_tokens"`
	RepeatStop        json.RawMessage   `json:"repeat_stop_tokens,omitempty"`
	AnthropicThinking string            `json:"anthropic_thinking,omitempty"`
	Sampling          json.RawMessage   `json:"sampling,omitempty"`
	ReasoningEffort   string            `json:"reasoning_effort,omitempty"`
	Vision            json.RawMessage   `json:"vision,omitempty"`
}

type standaloneVisionConfig struct {
	Exe       string `json:"exe"`
	MMProj    string `json:"mmproj"`
	Model     string `json:"model"`
	GPU       bool   `json:"gpu"`
	Threads   int    `json:"threads"`
	MaxTokens *int   `json:"max_tokens"`
	MinTokens int    `json:"min_tokens"`
}

func (c standaloneConfig) visionConfig(family requestnorm.Family) (*standaloneVisionConfig, []string, error) {
	if len(c.Vision) == 0 || bytes.Equal(bytes.TrimSpace(c.Vision), []byte("null")) {
		return nil, nil, nil
	}
	if family == requestnorm.FamilyDeepSeek {
		return nil, nil, errors.New("config vision is not supported for deepseek41")
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(c.Vision, &raw); err != nil || raw == nil {
		return nil, nil, errors.New("config vision must be an object")
	}
	allowed := map[string]bool{"exe": true, "mmproj": true, "model": true, "gpu": true, "threads": true, "max_tokens": true, "min_tokens": true}
	for key := range raw {
		if !allowed[key] {
			return nil, nil, fmt.Errorf("config vision contains unsupported field %q", key)
		}
	}
	var cfg standaloneVisionConfig
	if err := json.Unmarshal(c.Vision, &cfg); err != nil {
		return nil, nil, fmt.Errorf("invalid config vision: %w", err)
	}
	if cfg.Exe == "" || cfg.MMProj == "" || cfg.Model == "" {
		return nil, nil, errors.New("config vision requires exe, mmproj and model")
	}
	if cfg.Threads < 0 || cfg.MinTokens < 0 || (cfg.MaxTokens != nil && *cfg.MaxTokens < 0) {
		return nil, nil, errors.New("config vision token limits and threads must be non-negative")
	}
	absolute := func(path string) string {
		if !filepath.IsAbs(path) {
			path = filepath.Join(c.Cwd, path)
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return path
		}
		return abs
	}
	cfg.Exe, cfg.MMProj, cfg.Model = absolute(cfg.Exe), absolute(cfg.MMProj), absolute(cfg.Model)
	return &cfg, cfg.readerArgs(), nil
}

func (c standaloneVisionConfig) readerArgs() []string {
	args := []string{c.Exe, "--mmproj", c.MMProj, "--model", c.Model}
	if c.GPU {
		args = append(args, "--gpu")
	}
	if c.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(c.Threads))
	}
	maxTokens := 300
	if c.GPU {
		maxTokens = 1024
	}
	if c.MaxTokens != nil {
		maxTokens = *c.MaxTokens
	}
	if maxTokens > 0 {
		args = append(args, "--max-tokens", strconv.Itoa(maxTokens))
	}
	if c.MinTokens > 0 {
		args = append(args, "--min-tokens", strconv.Itoa(c.MinTokens))
	}
	return args
}

func (c standaloneConfig) aliasesConfig() ([]string, error) {
	if len(c.Aliases) == 0 || bytes.Equal(bytes.TrimSpace(c.Aliases), []byte("null")) {
		return nil, nil
	}
	var aliases []string
	var single string
	if err := json.Unmarshal(c.Aliases, &single); err == nil {
		aliases = strings.Split(single, ",")
	} else {
		var items []json.RawMessage
		if err := json.Unmarshal(c.Aliases, &items); err != nil {
			return nil, errors.New("aliases must be a list of model names or a comma-separated string")
		}
		for _, item := range items {
			if bytes.Equal(bytes.TrimSpace(item), []byte("null")) || json.Unmarshal(item, &single) != nil {
				return nil, errors.New("aliases must be a list of model names or a comma-separated string")
			}
			aliases = append(aliases, single)
		}
	}
	seen := map[string]bool{c.Model: true}
	out := make([]string, 0, len(aliases))
	for _, alias := range aliases {
		alias = strings.TrimSpace(alias)
		if alias != "" && !seen[alias] {
			seen[alias] = true
			out = append(out, alias)
		}
	}
	return out, nil
}

func (c standaloneConfig) repeatStopConfig() (*int, error) {
	if len(c.RepeatStop) == 0 {
		return nil, nil
	}
	value, err := requestnorm.ParseValue(c.RepeatStop)
	if err != nil {
		return nil, err
	}
	text, ok := value.Integer()
	if !ok {
		return nil, errors.New("repeat_stop_tokens must be a whole number >= 0")
	}
	n, err := strconv.Atoi(text)
	if err != nil {
		if strings.HasPrefix(text, "-") {
			return nil, errors.New("repeat_stop_tokens must be a whole number >= 0")
		}
		n = math.MaxInt
	}
	if n < 0 {
		return nil, errors.New("repeat_stop_tokens must be a whole number >= 0")
	}
	return &n, nil
}

func (c standaloneConfig) thinkingConfig() (*int, bool, error) {
	disabled := bytes.Equal(bytes.TrimSpace(c.AnswerRoom), []byte("false"))
	if len(c.ReasoningBudget) == 0 || bytes.Equal(bytes.TrimSpace(c.ReasoningBudget), []byte("null")) {
		return nil, disabled, nil
	}
	value, err := requestnorm.ParseValue(c.ReasoningBudget)
	if err != nil {
		return nil, false, err
	}
	var budget int
	if integer, ok := value.Integer(); ok {
		if !strings.HasPrefix(integer, "-") {
			budget, err = strconv.Atoi(integer)
			if err != nil {
				budget = math.MaxInt
			}
		}
	} else if number, ok := value.Float64(); ok && math.Trunc(number) == number {
		if number >= float64(math.MaxInt) {
			budget = math.MaxInt
		} else if number > 0 {
			budget = int(number)
		}
	} else {
		return nil, false, errors.New("reasoning_budget_tokens must be a whole number")
	}
	return &budget, disabled, nil
}

func readStandaloneConfig(path string) (standaloneConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return standaloneConfig{}, err
	}
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	var cfg standaloneConfig
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("invalid config: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return cfg, errors.New("invalid config: trailing JSON")
	} else if !errors.Is(err, io.EOF) {
		return cfg, fmt.Errorf("invalid config: %w", err)
	}
	if cfg.Exe == "" || cfg.Tokenizer == "" || cfg.Model == "" {
		return cfg, errors.New("config requires exe, tokenizer and model_name")
	}
	var args []string
	for _, arg := range cfg.Args {
		matches := joinedConfigArg.FindStringSubmatch(arg)
		if matches != nil && (len(args) == 0 || !bareConfigFlag.MatchString(args[len(args)-1])) {
			args = append(args, matches[1], strings.TrimSpace(matches[3]))
		} else {
			args = append(args, arg)
		}
	}
	cfg.Args = args
	if _, err := cfg.aliasesConfig(); err != nil {
		return cfg, fmt.Errorf("invalid config: %w", err)
	}
	if _, _, err := cfg.thinkingConfig(); err != nil {
		return cfg, fmt.Errorf("invalid config: %w", err)
	}
	if _, err := cfg.repeatStopConfig(); err != nil {
		return cfg, fmt.Errorf("invalid config: %w", err)
	}
	if cfg.AnthropicThinking != "" && cfg.AnthropicThinking != "model" && cfg.AnthropicThinking != "on_request" {
		return cfg, errors.New(`invalid config: anthropic_thinking must be "model" or "on_request"`)
	}
	switch cfg.ReasoningEffort {
	case "", "none", "low", "medium", "high":
	default:
		return cfg, errors.New(`invalid config: reasoning_effort must be none, low, medium or high`)
	}
	if !filepath.IsAbs(cfg.Exe) {
		cfg.Exe, err = filepath.Abs(filepath.Join(cfg.Cwd, cfg.Exe))
		if err != nil {
			return cfg, err
		}
	}
	return cfg, nil
}

func (c standaloneConfig) childEnv() ([]string, error) {
	env := os.Environ()
	if c.GPU != nil {
		if *c.GPU < 0 {
			return nil, errors.New("config gpu must be non-negative")
		}
		env = append(env, "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES="+strconv.Itoa(*c.GPU))
	}
	for key, value := range c.Env {
		if key == "" || strings.ContainsAny(key, "=\x00") || strings.ContainsRune(value, '\x00') {
			return nil, errors.New("invalid config environment")
		}
		env = append(env, key+"="+value)
	}
	return env, nil
}
