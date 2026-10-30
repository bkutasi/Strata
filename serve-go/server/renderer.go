package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
	"strata/template"
)

type Renderer interface {
	Render(context.Context, requestnorm.Result) (Rendered, error)
}

type Rendered struct {
	Prompt   string
	Plain    []modeltokenizer.Span
	Thinking bool
}

// TemplateRenderer is restricted to trusted model pack templates. Until literal
// marking is ported, requests containing a control-token literal are rejected.
type TemplateRenderer struct {
	renderer *template.Renderer
	literals []string
	mu       sync.Mutex
}

func NewTemplateRenderer(pack string, tokenizer *modeltokenizer.Tokenizer) (*TemplateRenderer, error) {
	if tokenizer == nil {
		return nil, errors.New("missing tokenizer")
	}
	dir := filepath.Join(pack, "tokenizer")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		dir = pack
	} else if err != nil {
		return nil, err
	}
	source, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja"))
	if err != nil {
		return nil, err
	}
	t, err := template.New(string(source))
	if err != nil {
		return nil, err
	}
	literals := []string{"<think>", "</think>"}
	literals = append(literals, tokenizer.ControlLiterals()...)
	return &TemplateRenderer{renderer: t, literals: literals}, nil
}

func (r *TemplateRenderer) Render(ctx context.Context, input requestnorm.Result) (Rendered, error) {
	messages, err := json.Marshal(input.Messages)
	if err != nil {
		return Rendered{}, err
	}
	tools, err := json.Marshal(input.Tools)
	if err != nil {
		return Rendered{}, err
	}
	kwargs, err := json.Marshal(input.Kwargs)
	if err != nil {
		return Rendered{}, err
	}
	for _, data := range [][]byte{messages, tools, kwargs} {
		var v any
		if err = json.Unmarshal(data, &v); err != nil {
			return Rendered{}, err
		}
		if r.hasLiteral(v) {
			return Rendered{}, errors.New("control-token literals in request are not supported")
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	prompt, err := r.renderer.Render(ctx, template.Input{Messages: messages, Tools: tools, Kwargs: kwargs, AddGenerationPrompt: true})
	if err != nil {
		return Rendered{}, fmt.Errorf("render template: %w", err)
	}
	thinking := true
	if v, ok := input.Kwargs.Lookup("enable_thinking"); ok {
		if b, ok := v.Bool(); ok {
			thinking = b
		}
	}
	return Rendered{Prompt: prompt, Thinking: thinking}, nil
}

func (r *TemplateRenderer) hasLiteral(v any) bool {
	switch x := v.(type) {
	case string:
		for _, tag := range r.literals {
			if strings.Contains(x, tag) {
				return true
			}
		}
	case []any:
		for _, item := range x {
			if r.hasLiteral(item) {
				return true
			}
		}
	case map[string]any:
		for k, item := range x {
			if r.hasLiteral(k) || r.hasLiteral(item) {
				return true
			}
		}
	}
	return false
}
