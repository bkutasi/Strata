package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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

// TemplateRenderer is restricted to trusted model pack templates.
type TemplateRenderer struct {
	renderer *template.Renderer
	literals []string
	source   string
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
	return &TemplateRenderer{renderer: t, literals: literalTags(tokenizer.ControlLiterals()), source: string(source)}, nil
}

func (r *TemplateRenderer) Render(ctx context.Context, input requestnorm.Result) (Rendered, error) {
	return r.render(ctx, input.Messages, input.Tools, input.Kwargs)
}

func (r *TemplateRenderer) RenderAnthropic(ctx context.Context, input requestnorm.AnthropicResult) (Rendered, error) {
	return r.render(ctx, input.Messages, input.Tools, input.Kwargs)
}

func (r *TemplateRenderer) render(ctx context.Context, inputMessages any, inputTools []requestnorm.Value, inputKwargs requestnorm.Value) (Rendered, error) {
	messages, err := json.Marshal(inputMessages)
	if err != nil {
		return Rendered{}, err
	}
	tools, err := json.Marshal(inputTools)
	if err != nil {
		return Rendered{}, err
	}
	kwargs, err := json.Marshal(inputKwargs)
	if err != nil {
		return Rendered{}, err
	}
	marker, err := newLiteralMarker(r.literals, r.source, messages, tools, kwargs)
	if err != nil {
		return Rendered{}, err
	}
	if marker != nil {
		messages, tools, kwargs, err = marker.markInputs()
		if err != nil {
			return Rendered{}, err
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	prompt, err := r.renderer.Render(ctx, template.Input{Messages: messages, Tools: tools, Kwargs: kwargs, AddGenerationPrompt: true})
	if err != nil {
		return Rendered{}, fmt.Errorf("render template: %w", err)
	}
	thinking := true
	if v, ok := inputKwargs.Lookup("enable_thinking"); ok {
		if b, ok := v.Bool(); ok {
			thinking = b
		}
	}
	var plain []modeltokenizer.Span
	if marker != nil {
		prompt, plain = marker.restore(prompt)
	}
	return Rendered{Prompt: prompt, Plain: plain, Thinking: thinking}, nil
}
