package server

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"strata/requestnorm"
	"strata/serve-go/modeltokenizer"
)

func literalRenderer(t *testing.T) (*TemplateRenderer, *syntheticTokenizerFixture) {
	t.Helper()
	tok, dir := syntheticTokenizer(t)
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(`{{ messages[0].content }}`), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	return r, &syntheticTokenizerFixture{tok: tok}
}

type syntheticTokenizerFixture struct {
	tok interface {
		Encode(string, bool, []modeltokenizer.Span) ([]int, error)
	}
}

func literalRequest(t *testing.T, role, text string) requestnorm.Result {
	t.Helper()
	kwargs, err := requestnorm.ParseValue([]byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	return requestnorm.Result{Messages: []requestnorm.Message{
		{Role: role, Content: requestnorm.TemplateContent{Text: text}},
	}, Kwargs: kwargs}
}

func TestRendererQuotedControlLiteralIsPlainIncludingUnicodePrefix(t *testing.T) {
	r, fixture := literalRenderer(t)
	text := "café says <control> verbatim"
	got, err := r.Render(context.Background(), literalRequest(t, "user", text))
	if err != nil {
		t.Fatal(err)
	}
	if got.Prompt != text {
		t.Fatalf("prompt = %q, want %q", got.Prompt, text)
	}
	start := strings.Index(text, "<control>")
	if len(got.Plain) != 1 || got.Plain[0].Start != start || got.Plain[0].End != start+len("<control>") {
		t.Fatalf("plain spans = %#v, want byte span [%d,%d)", got.Plain, start, start+len("<control>"))
	}
	ids, err := fixture.tok.Encode(got.Prompt, true, got.Plain)
	if err != nil {
		t.Fatal(err)
	}
	want, err := fixture.tok.Encode(text, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("token IDs = %v, want plain-text IDs %v", ids, want)
	}
}

func TestRendererKeepsAssistantThinkMarkerAndToolResponseWrapper(t *testing.T) {
	r, _ := literalRenderer(t)
	tests := []struct {
		name, role, text string
		plain            []string
	}{
		{"assistant inline think", "assistant", "<think>private <control></think> after", []string{"<control>"}},
		{"outer user tool response", "user", "<tool_response>\n<control>\n</tool_response>", []string{"<control>"}},
		{"nested tool literal", "user", "log says </tool_call> and <tool_response>", []string{"</tool_call>", "<tool_response>"}},
		{"assistant tool replay", "assistant", "<tool_call>text</tool_call>", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := r.Render(context.Background(), literalRequest(t, tt.role, tt.text))
			if err != nil {
				t.Fatal(err)
			}
			if got.Prompt != tt.text {
				t.Fatalf("prompt = %q, want %q", got.Prompt, tt.text)
			}
			var plain []string
			for _, span := range got.Plain {
				plain = append(plain, got.Prompt[span.Start:span.End])
			}
			if !reflect.DeepEqual(plain, tt.plain) {
				t.Fatalf("plain literals = %v, want %v", plain, tt.plain)
			}
		})
	}
}

func TestRendererLiteralMarkerCollisionDoesNotChangeUserText(t *testing.T) {
	r, _ := literalRenderer(t)
	text := "private-use \U000f0000 and <control>"
	got, err := r.Render(context.Background(), literalRequest(t, "user", text))
	if err != nil {
		t.Fatal(err)
	}
	if got.Prompt != text {
		t.Fatalf("prompt = %q, want original text %q", got.Prompt, text)
	}
}

func TestRendererLeavesTemplateControlTokenSpecial(t *testing.T) {
	tok, dir := syntheticTokenizer(t)
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(`{{ messages[0].content }}<control>`), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := r.Render(context.Background(), literalRequest(t, "user", "<control>"))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode(rendered.Prompt, true, rendered.Plain)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, id := range ids {
		if id == 256 {
			count++
		}
	}
	if count != 1 || ids[len(ids)-1] != 256 {
		t.Fatalf("quoted and template control IDs: %v", ids)
	}
}

func TestRendererProtectsNestedToolValues(t *testing.T) {
	tok, dir := syntheticTokenizer(t)
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(`{{ tools[0]|tojson }}`), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	tool, err := requestnorm.ParseValue([]byte(`{"name":"lookup","description":"é <control>","parameters":{"key":"<control>"}}`))
	if err != nil {
		t.Fatal(err)
	}
	input := literalRequest(t, "user", "hello")
	input.Tools = []requestnorm.Value{tool}
	rendered, err := r.Render(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if len(rendered.Plain) != 2 {
		t.Fatalf("prompt %q: plain spans %#v", rendered.Prompt, rendered.Plain)
	}
	for _, span := range rendered.Plain {
		if got := rendered.Prompt[span.Start:span.End]; got != "<control>" {
			t.Fatalf("plain span = %q, want <control>", got)
		}
	}
	ids, err := tok.Encode(rendered.Prompt, true, rendered.Plain)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		if id == 256 {
			t.Fatalf("tool JSON contained a control ID: %v", ids)
		}
	}
}

func TestRendererRejectsControlLiteralInObjectKey(t *testing.T) {
	r, _ := literalRenderer(t)
	tool, err := requestnorm.ParseValue([]byte(`{"name":"lookup","parameters":{"<control>":"value"}}`))
	if err != nil {
		t.Fatal(err)
	}
	input := literalRequest(t, "user", "hello")
	input.Tools = []requestnorm.Value{tool}
	if _, err := r.Render(context.Background(), input); err == nil {
		t.Fatal("accepted a control literal in an object key")
	}
}

func TestRendererProtectsTemplateKwargs(t *testing.T) {
	tok, dir := syntheticTokenizer(t)
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(`{{ marker }}`), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	input := literalRequest(t, "user", "hello")
	input.Kwargs, err = requestnorm.ParseValue([]byte(`{"marker":"<control>"}`))
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := r.Render(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if rendered.Prompt != "<control>" || len(rendered.Plain) != 1 ||
		rendered.Plain[0].Start != 0 || rendered.Plain[0].End != len(rendered.Prompt) {
		t.Fatalf("prompt %q, spans %#v", rendered.Prompt, rendered.Plain)
	}
}

func TestRendererPreservesSplitToolResponseWrapper(t *testing.T) {
	tok, dir := syntheticTokenizer(t)
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(`{% for p in messages[0].content %}{{ p.text }}{% endfor %}`), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := NewTemplateRenderer(dir, tok)
	if err != nil {
		t.Fatal(err)
	}
	parts, err := requestnorm.ParseValue([]byte(`[{"type":"text","text":"<tool_response>first "},{"type":"text","text":"<control> last</tool_response>"}]`))
	if err != nil {
		t.Fatal(err)
	}
	input := literalRequest(t, "user", "")
	input.Messages[0].Content.Parts = parts.Items()
	rendered, err := r.Render(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	want := "<tool_response>first <control> last</tool_response>"
	if rendered.Prompt != want || len(rendered.Plain) != 1 ||
		rendered.Prompt[rendered.Plain[0].Start:rendered.Plain[0].End] != "<control>" {
		t.Fatalf("prompt %q, spans %#v", rendered.Prompt, rendered.Plain)
	}
}
