package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// standaloneConfig supports the fields in the standalone Strata run config.
// Unknown settings are rejected rather than silently ignored.
type standaloneConfig struct {
	Exe       string            `json:"exe"`
	Cwd       string            `json:"cwd"`
	Args      []string          `json:"args"`
	Tokenizer string            `json:"tokenizer"`
	Model     string            `json:"model_name"`
	Host      string            `json:"host"`
	APIKey    string            `json:"api_key"`
	GPU       *int              `json:"gpu"`
	Env       map[string]string `json:"env"`
	Log       string            `json:"log"`
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
