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
	"strconv"
	"strings"

	"strata/requestnorm"
)

// standaloneConfig supports the fields in the standalone Strata run config.
// Unknown settings are rejected rather than silently ignored.
type standaloneConfig struct {
	Exe               string            `json:"exe"`
	Cwd               string            `json:"cwd"`
	Args              []string          `json:"args"`
	Tokenizer         string            `json:"tokenizer"`
	Model             string            `json:"model_name"`
	Host              string            `json:"host"`
	APIKey            string            `json:"api_key"`
	GPU               *int              `json:"gpu"`
	Env               map[string]string `json:"env"`
	Log               string            `json:"log"`
	AnswerRoom        json.RawMessage   `json:"answer_room"`
	ReasoningBudget   json.RawMessage   `json:"reasoning_budget_tokens"`
	RepeatStop        json.RawMessage   `json:"repeat_stop_tokens,omitempty"`
	AnthropicThinking string            `json:"anthropic_thinking,omitempty"`
	Sampling          json.RawMessage   `json:"sampling,omitempty"`
	ReasoningEffort   string            `json:"reasoning_effort,omitempty"`
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
