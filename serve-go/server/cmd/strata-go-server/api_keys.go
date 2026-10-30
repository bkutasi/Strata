package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
)

func splitAPIKeys(value string) ([]string, error) {
	if value == "" {
		return nil, nil
	}
	if strings.EqualFold(strings.TrimSpace(value), "none") {
		return nil, nil
	}
	var keys []string
	for _, key := range strings.Split(value, ",") {
		if key = strings.TrimSpace(key); key != "" {
			keys = append(keys, key)
		}
	}
	if len(keys) == 0 {
		return nil, errors.New("API key is empty")
	}
	return keys, nil
}

func configAPIKeys(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		return splitAPIKeys(value)
	}
	var keys []string
	if err := json.Unmarshal(raw, &keys); err != nil || len(keys) == 0 {
		return nil, errors.New("api_key must be a string or a nonempty list of strings")
	}
	for i, key := range keys {
		keys[i] = strings.TrimSpace(key)
		if keys[i] == "" {
			return nil, errors.New("API key is empty")
		}
	}
	return keys, nil
}
