package server

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type logprobChoice struct {
	Message struct {
		Content   *string `json:"content"`
		Reasoning string  `json:"reasoning_content"`
	} `json:"message"`
	Delta struct {
		Content   string `json:"content"`
		Reasoning string `json:"reasoning_content"`
	} `json:"delta"`
	Logprobs struct {
		Content   []logprobEntry `json:"content"`
		Reasoning []logprobEntry `json:"reasoning_content"`
	} `json:"logprobs"`
}

func readLogprobReply(t *testing.T, mode, fields string, stream bool) (string, string, []logprobEntry, []logprobEntry) {
	t.Helper()
	h, _ := setupEngine(t, mode)
	ts := httptest.NewServer(h)
	defer ts.Close()
	streamValue := "false"
	if stream {
		streamValue = "true"
	}
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":100,"stream":` + streamValue + `,` + fields + `}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/chat/completions", body, "secret")
	if status != http.StatusOK {
		t.Fatalf("status %d, body %s", status, raw)
	}
	var content, reasoning strings.Builder
	var contentLP, reasonLP []logprobEntry
	lines := []string{raw}
	if stream {
		lines = strings.Split(raw, "\n")
	}
	for _, line := range lines {
		if stream {
			if !strings.HasPrefix(line, "data: {") {
				continue
			}
			line = strings.TrimPrefix(line, "data: ")
		}
		var response struct {
			Choices []logprobChoice `json:"choices"`
		}
		if err := json.Unmarshal([]byte(line), &response); err != nil || len(response.Choices) != 1 {
			t.Fatalf("invalid response: %s (%v)", line, err)
		}
		choice := response.Choices[0]
		if stream {
			content.WriteString(choice.Delta.Content)
			reasoning.WriteString(choice.Delta.Reasoning)
		} else {
			if choice.Message.Content != nil {
				content.WriteString(*choice.Message.Content)
			}
			reasoning.WriteString(choice.Message.Reasoning)
		}
		contentLP = append(contentLP, choice.Logprobs.Content...)
		reasonLP = append(reasonLP, choice.Logprobs.Reasoning...)
	}
	return content.String(), reasoning.String(), contentLP, reasonLP
}

func spellLogprobs(entries []logprobEntry) string {
	var text strings.Builder
	for _, entry := range entries {
		text.WriteString(entry.Token)
	}
	return text.String()
}

func TestChatLogprobsHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		content, _, entries, _ := readLogprobReply(t, "engine-logprobs",
			`"logprobs":true,"top_logprobs":2,"allowed_token_ids":[65,72],"chat_template_kwargs":{"enable_thinking":false}`, stream)
		if content != "Café costs €5." || spellLogprobs(entries) != content || len(entries) != len([]byte(content)) {
			t.Fatalf("stream=%v content=%q entries=%#v", stream, content, entries)
		}
		for _, entry := range entries {
			if len(entry.TopLogprobs) != 2 || math.Abs(math.Exp(entry.TopLogprobs[0].Logprob)+math.Exp(entry.TopLogprobs[1].Logprob)-1) > 1e-9 {
				t.Fatalf("stream=%v alternatives=%#v", stream, entry.TopLogprobs)
			}
		}
		content, _, entries, _ = readLogprobReply(t, "engine-logprobs",
			`"logprobs":true,"stop":["5.!"],"chat_template_kwargs":{"enable_thinking":false}`, stream)
		if content != "Café costs €5." || spellLogprobs(entries) != content {
			t.Fatalf("held stop stream=%v content=%q entries=%#v", stream, content, entries)
		}
	}
}

func TestChatStructuredLogprobsHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		content, _, entries, _ := readLogprobReply(t, "engine-logprobs-json",
			`"logprobs":true,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":false}`, stream)
		if content != `{"x": 1}` || spellLogprobs(entries) != content {
			t.Fatalf("stream=%v content=%q entries=%#v", stream, content, entries)
		}
	}
}

func TestChatReasoningLogprobsHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		content, reasoning, entries, reasonEntries := readLogprobReply(t, "engine-logprobs-thinking",
			`"logprobs":true,"chat_template_kwargs":{"enable_thinking":true}`, stream)
		if content != "answer" || reasoning != "think" || spellLogprobs(entries) != content || spellLogprobs(reasonEntries) != reasoning {
			t.Fatalf("stream=%v content=%q reasoning=%q entries=%#v reasoning entries=%#v", stream, content, reasoning, entries, reasonEntries)
		}
	}
}

func TestChatRewrittenStructuredLogprobsHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		content, reasoning, entries, reasonEntries := readLogprobReply(t, "engine-logprobs-repair",
			`"logprobs":true,"response_format":{"type":"json_object","repair_latex_escapes":true},"chat_template_kwargs":{"enable_thinking":true}`, stream)
		if content != `{"x":"\\q"}` || reasoning != "think" || len(entries) != 0 || len(reasonEntries) != 0 {
			t.Fatalf("stream=%v content=%q reasoning=%q entries=%#v reasoning entries=%#v", stream, content, reasoning, entries, reasonEntries)
		}
	}
}

func TestChatLogprobsOldEngineStopHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		content, _, entries, _ := readLogprobReply(t, "engine-logprobs-old-stop",
			`"logprobs":true,"stop":"A","chat_template_kwargs":{"enable_thinking":false}`, stream)
		if content != "" || len(entries) != 0 {
			t.Fatalf("stream=%v content=%q entries=%#v", stream, content, entries)
		}
	}
}
