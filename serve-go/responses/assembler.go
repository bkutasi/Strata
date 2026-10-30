package responses

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"strata/requestnorm"
)

// ParserEvent is the model output parser's event. Call arguments for a final
// tool_call must be an ordered JSON object. tool_start and tool_args are
// provisional and must not be executed.
type ParserEvent struct {
	Kind string
	Text string
	Call *Call
}

type Call struct {
	ID        string
	Name      string
	Arguments requestnorm.Value
}

// Event is a Responses event ready for JSON or SSE serialization.
type Event map[string]any

// AssemblerOptions supplies per-generation dependencies. Validate is required
// for JSON formats; it must perform the serving path's structured-output
// validation and return the final text, or an error.
type AssemblerOptions struct {
	Model        string
	PromptTokens int64
	ID           func(prefix string) (string, error)
	Now          func() time.Time
	Validate     func(text, finish string) (string, error)
}

// Done carries the engine's final accounting. Nil PromptTokens uses the
// count supplied at construction. Timings is copied into the final event.
type Done struct {
	Finish           string
	CompletionTokens int64
	PromptTokens     *int64
	Reused           int64
	ReasoningTokens  int64
	Timings          map[string]any
}

// Assembler is single-use and must not be accessed concurrently. Returned
// events are snapshots; modifying one cannot mutate the assembler.
type Assembler struct {
	response  map[string]any
	names     map[string]ToolName
	item      map[string]any
	id        func(string) (string, error)
	now       func() time.Time
	validate  func(string, string) (string, error)
	prompt    int64
	seq       int
	jsonMode  bool
	encrypted bool
	called    bool
	jsonText  strings.Builder
	started   bool
	terminal  bool
}

func randomID(prefix string) (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return prefix + "_" + hex.EncodeToString(b[:]), nil
}

func valueAny(v requestnorm.Value) any {
	b, err := v.MarshalJSON()
	if err != nil {
		panic(err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		panic(err)
	}
	return out
}

func clone(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, v := range x {
			out[k] = clone(v)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, v := range x {
			out[i] = clone(v)
		}
		return out
	default:
		return x
	}
}

// NewAssembler creates an assembler for a request already checked by Prepare.
// Prepared.Names must correspond to that request's tools.
func NewAssembler(req *Request, prepared Prepared, options AssemblerOptions) (*Assembler, error) {
	if req == nil {
		return nil, invalid("", "request is nil")
	}
	if err := checkRequest(req.root); err != nil {
		return nil, err
	}
	if options.Model == "" || !utf8.ValidString(options.Model) || options.PromptTokens < 0 {
		return nil, invalid("", "model must be non-empty and prompt tokens non-negative")
	}
	format, err := textFormat(req.root)
	if err != nil {
		return nil, err
	}
	if format.Kind() != requestnorm.Null && options.Validate == nil {
		return nil, fmt.Errorf("%w: structured output requires a validator", ErrUnsupported)
	}
	id := options.ID
	if id == nil {
		id = randomID
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	responseID, err := id("resp")
	if err != nil {
		return nil, err
	}
	if responseID == "" || !utf8.ValidString(responseID) {
		return nil, invalid("", "ID generator returned an invalid ID")
	}
	root := req.root
	reasoning := field(root, "reasoning")
	choice := any("auto")
	if v, present := root.Lookup("tool_choice"); present {
		choice = valueAny(v)
	}
	parallel := any(true)
	if v, present := root.Lookup("parallel_tool_calls"); present {
		parallel = valueAny(v)
	}
	text := any(map[string]any{"format": map[string]any{"type": "text"}})
	if v := field(root, "text"); v.Kind() == requestnorm.Object {
		text = valueAny(v)
	}
	tools := any([]any{})
	if v := field(root, "tools"); v.Kind() == requestnorm.Array && len(v.Items()) > 0 {
		tools = valueAny(v)
	}
	metadata := any(map[string]any{})
	if v := field(root, "metadata"); v.Kind() == requestnorm.Object && len(v.Fields()) > 0 {
		metadata = valueAny(v)
	}
	names := make(map[string]ToolName, len(prepared.Names))
	for k, v := range prepared.Names {
		names[k] = v
	}
	encrypted := false
	for _, v := range field(root, "include").Items() {
		if s, _ := stringOf(v); s == "reasoning.encrypted_content" {
			encrypted = true
		}
	}
	return &Assembler{
		id: id, now: now, names: names, validate: options.Validate, prompt: options.PromptTokens,
		jsonMode: format.Kind() != requestnorm.Null, encrypted: encrypted,
		response: map[string]any{
			"id": responseID, "object": "response", "created_at": now().Unix(), "status": "in_progress",
			"background": false, "error": nil, "incomplete_details": nil,
			"instructions": valueAny(field(root, "instructions")), "max_output_tokens": valueAny(field(root, "max_output_tokens")),
			"model": options.Model, "output": []any{}, "parallel_tool_calls": parallel, "previous_response_id": nil,
			"reasoning": map[string]any{"effort": valueAny(field(reasoning, "effort")), "summary": valueAny(field(reasoning, "summary"))},
			"store":     false, "temperature": valueAny(field(root, "temperature")), "text": text, "tool_choice": choice,
			"tools": tools, "top_p": valueAny(field(root, "top_p")), "truncation": "disabled", "usage": nil, "user": nil,
			"metadata": metadata,
		},
	}, nil
}

func (a *Assembler) event(kind string, data map[string]any) Event {
	e := Event{"type": kind, "sequence_number": a.seq}
	a.seq++
	for k, v := range data {
		e[k] = clone(v)
	}
	return e
}

func (a *Assembler) index() int { return len(a.response["output"].([]any)) - 1 }

func (a *Assembler) Start() ([]Event, error) {
	if a.started || a.terminal {
		return nil, errors.New("responses: assembler already started or closed")
	}
	a.started = true
	return []Event{a.event("response.created", map[string]any{"response": a.response}),
		a.event("response.in_progress", map[string]any{"response": a.response})}, nil
}

func (a *Assembler) ready() error {
	if !a.started || a.terminal {
		return errors.New("responses: assembler not started or already closed")
	}
	return nil
}

func (a *Assembler) InProgress() (Event, error) {
	if err := a.ready(); err != nil {
		return nil, err
	}
	return a.event("response.in_progress", map[string]any{"response": a.response}), nil
}

func (a *Assembler) makeID(prefix string) (string, error) {
	id, err := a.id(prefix)
	if err != nil {
		return "", err
	}
	if id == "" || !utf8.ValidString(id) {
		return "", invalid("", "ID generator returned an invalid ID")
	}
	return id, nil
}

func (a *Assembler) open(item map[string]any) []Event {
	a.response["output"] = append(a.response["output"].([]any), item)
	a.item = item
	out := []Event{a.event("response.output_item.added", map[string]any{"output_index": a.index(), "item": item})}
	if item["type"] == "message" {
		part := map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}}
		item["content"] = append(item["content"].([]any), part)
		out = append(out, a.event("response.content_part.added", map[string]any{
			"item_id": item["id"], "output_index": a.index(), "content_index": 0, "part": part}))
	}
	return out
}

func (a *Assembler) close(status string) []Event {
	item := a.item
	if item == nil {
		return nil
	}
	a.item = nil
	at := a.index()
	kind := item["type"].(string)
	if status == "" {
		status = "completed"
		if kind == "function_call" || kind == "custom_tool_call" {
			status = "incomplete"
		}
	}
	item["status"] = status
	out := []Event{}
	switch kind {
	case "message":
		part := item["content"].([]any)[0].(map[string]any)
		out = append(out, a.event("response.output_text.done", map[string]any{"item_id": item["id"],
			"output_index": at, "content_index": 0, "text": part["text"], "logprobs": []any{}}))
		out = append(out, a.event("response.content_part.done", map[string]any{"item_id": item["id"],
			"output_index": at, "content_index": 0, "part": part}))
	case "reasoning":
		text := item["content"].([]any)[0].(map[string]any)["text"].(string)
		out = append(out, a.event("response.reasoning_text.done", map[string]any{
			"item_id": item["id"], "output_index": at, "content_index": 0, "text": text}))
		if a.encrypted {
			payload, _ := pythonJSON(object(f("text", strValue(text))))
			item["encrypted_content"] = EncryptedPrefix + base64.StdEncoding.EncodeToString([]byte(payload))
		}
	case "function_call":
		out = append(out, a.event("response.function_call_arguments.done", map[string]any{
			"item_id": item["id"], "output_index": at, "name": item["name"], "arguments": item["arguments"]}))
	case "custom_tool_call":
		out = append(out, a.event("response.custom_tool_call_input.done", map[string]any{
			"item_id": item["id"], "output_index": at, "input": item["input"]}))
	}
	return append(out, a.event("response.output_item.done", map[string]any{"output_index": at, "item": item}))
}

func (a *Assembler) callItem(call *Call) (map[string]any, error) {
	mapping, found := a.names[call.Name]
	if !found {
		mapping = ToolName{Name: call.Name, Kind: "function"}
	}
	prefix, kind, key := "fc", "function_call", "arguments"
	if mapping.Kind == "custom" {
		prefix, kind, key = "ctc", "custom_tool_call", "input"
	}
	id, err := a.makeID(prefix)
	if err != nil {
		return nil, err
	}
	item := map[string]any{"id": id, "type": kind, "status": "in_progress", "call_id": call.ID,
		"name": mapping.Name, key: ""}
	if mapping.Namespace != "" {
		item["namespace"] = mapping.Namespace
	}
	return item, nil
}

func (a *Assembler) release(text string) ([]Event, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	id, err := a.makeID("msg")
	if err != nil {
		return nil, err
	}
	out := a.open(map[string]any{"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}})
	a.item["content"].([]any)[0].(map[string]any)["text"] = text
	out = append(out, a.event("response.output_text.delta", map[string]any{"item_id": id,
		"output_index": a.index(), "content_index": 0, "delta": text, "logprobs": []any{}}))
	return append(out, a.close("")...), nil
}

func (a *Assembler) callStarted() ([]Event, error) {
	var out []Event
	if a.jsonMode && !a.called {
		var err error
		out, err = a.release(a.jsonText.String())
		if err != nil {
			return out, err
		}
		a.jsonText.Reset()
	}
	a.called = true
	return out, nil
}
