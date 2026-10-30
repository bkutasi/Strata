// Package modeltokenizer joins pack metadata, pre-tokenization and byte BPE.
package modeltokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"strata/serve-go/bpe"
	"strata/tokenizer"
)

// Span protects literal starts in [Start, End), using UTF-8 byte offsets.
type Span = bpe.Span

// Tokenizer is immutable and may be used concurrently.
type Tokenizer struct {
	core       *bpe.Tokenizer
	specialIDs map[string]int
}

// Load reads a pack root containing tokenizer/, or its tokenizer directory.
// Missing or malformed tokenizer.json and Python-falsy pre values default to
// qwen35, as in the Python server. Unknown nonempty names are rejected.
func Load(path string) (*Tokenizer, error) {
	dir := filepath.Join(path, "tokenizer")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		dir = path
	} else if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	var cfg struct {
		Model      string                     `json:"model"`
		Pre        json.RawMessage            `json:"pre"`
		SpecialIDs map[string]json.RawMessage `json:"special_ids"`
	}
	if len(data) != 0 && json.Valid(data) {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("tokenizer.json: %w", err)
		}
	}
	// DeepSeek packs currently omit model, but are byte-level BPE.
	if cfg.Model != "" && cfg.Model != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer model %q", cfg.Model)
	}
	var pre string
	if len(cfg.Pre) != 0 {
		var value any
		if err := json.Unmarshal(cfg.Pre, &value); err != nil {
			return nil, fmt.Errorf("invalid tokenizer pre: %w", err)
		}
		switch v := value.(type) {
		case nil:
		case string:
			pre = v
		case bool:
			if v {
				return nil, fmt.Errorf("invalid tokenizer pre: expected a family name")
			}
		case float64:
			if v != 0 {
				return nil, fmt.Errorf("invalid tokenizer pre: expected a family name")
			}
		case []any:
			if len(v) != 0 {
				return nil, fmt.Errorf("invalid tokenizer pre: expected a family name")
			}
		case map[string]any:
			if len(v) != 0 {
				return nil, fmt.Errorf("invalid tokenizer pre: expected a family name")
			}
		}
	}
	var split func(string) ([]tokenizer.Span, error)
	switch pre {
	case "", "qwen35":
		split = tokenizer.Qwen35
	case "deepseek3", "deepseek-v3":
		split = tokenizer.DeepSeek3
	default:
		return nil, fmt.Errorf("unsupported tokenizer pre %q", pre)
	}
	core, err := bpe.Load(dir, func(text string) []string {
		// BPE validates UTF-8 before invoking this splitter.
		spans, err := split(text)
		if err != nil {
			return nil
		}
		pieces := make([]string, len(spans))
		for i, span := range spans {
			pieces[i] = text[span.Start:span.End]
		}
		return pieces
	})
	if err != nil {
		return nil, err
	}
	specialIDs := make(map[string]int, len(cfg.SpecialIDs))
	if len(cfg.SpecialIDs) != 0 {
		data, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
		if err != nil {
			return nil, err
		}
		var vocab map[string]json.RawMessage
		if err := json.Unmarshal(data, &vocab); err != nil {
			return nil, fmt.Errorf("vocab.json: %w", err)
		}
		for name, raw := range cfg.SpecialIDs {
			var id int
			if string(raw) == "null" {
				return nil, fmt.Errorf("special token %q has null id", name)
			}
			if err := json.Unmarshal(raw, &id); err != nil {
				return nil, fmt.Errorf("special token %q: %w", name, err)
			}
			if id < 0 || id >= len(vocab) {
				return nil, fmt.Errorf("special token %q has out-of-range id %d", name, id)
			}
			specialIDs[name] = id
		}
	}
	return &Tokenizer{core: core, specialIDs: specialIDs}, nil
}

// SpecialIDs returns a copy of the pack's special_ids, preserving its names.
// Qwen packs use GGUF keys such as tokenizer.ggml.eos_token_id; DeepSeek
// packs use short names such as eos. Missing metadata yields an empty map.
func (t *Tokenizer) SpecialIDs() map[string]int {
	ids := make(map[string]int, len(t.specialIDs))
	for name, id := range t.specialIDs {
		ids[name] = id
	}
	return ids
}

// Encode tokenizes text. Protected spans are byte offsets in the original text.
// Use EncodeWithBudget for untrusted API input.
func (t *Tokenizer) Encode(text string, parseSpecial bool, plain []Span) ([]int, error) {
	return t.core.Encode(text, parseSpecial, plain)
}

// EncodeWithBudget rejects input larger than maxInputBytes before tokenization.
// This limits UTF-8 input bytes, not memory consumption or splitter allocations.
func (t *Tokenizer) EncodeWithBudget(text string, parseSpecial bool, plain []Span, maxInputBytes int) ([]int, error) {
	return t.core.EncodeWithBudget(text, parseSpecial, plain, maxInputBytes)
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) (string, error) { return t.core.Decode(ids) }

// TokenBytes returns the raw bytes represented by a token ID.
func (t *Tokenizer) TokenBytes(id int) ([]byte, error) { return t.core.TokenBytes(id) }
