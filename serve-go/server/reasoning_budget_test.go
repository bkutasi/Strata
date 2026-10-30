package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestReasoningBudgetRequestCompatibility(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	for _, api := range []struct {
		path, disabled, enabled string
	}{
		{"/v1/chat/completions", `"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"enable_thinking":false}`, `"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"enable_thinking":true}`},
		{"/v1/messages", `"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"disabled"}`, `"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"enabled"}`},
		{"/v1/responses", `"input":"Hi","reasoning":{"effort":"none"}`, `"input":"Hi","reasoning":{"effort":"low"}`},
	} {
		for _, tc := range []struct {
			fields string
			think  bool
			status int
		}{
			{`"reasoning_budget_tokens":0`, true, http.StatusOK},
			{`"reasoning_budget_tokens":-3`, true, http.StatusOK},
			{`"reasoning_budget_tokens":0.0`, true, http.StatusOK},
			{`"thinking_budget_tokens":0`, true, http.StatusOK},
			{`"reasoning_budget_tokens":null,"thinking_budget_tokens":0`, true, http.StatusOK},
			{`"reasoning_budget_tokens":0,"thinking_budget_tokens":12`, true, http.StatusOK},
			{`"reasoning_budget_tokens":0,"thinking_budget_tokens":"ignored"`, true, http.StatusOK},
			{`"reasoning_budget_tokens":12`, false, http.StatusOK},
			{`"thinking_budget_tokens":12`, false, http.StatusOK},
			{`"reasoning_budget_tokens":12`, true, http.StatusBadRequest},
			{`"reasoning_budget_tokens":null,"thinking_budget_tokens":12`, true, http.StatusBadRequest},
			{`"reasoning_budget_tokens":1.5`, false, http.StatusBadRequest},
			{`"reasoning_budget_tokens":true`, false, http.StatusBadRequest},
			{`"reasoning_budget_tokens":"0"`, false, http.StatusBadRequest},
		} {
			prompt := api.disabled
			if tc.think {
				prompt = api.enabled
			}
			for _, stream := range []string{"false", "true"} {
				body := `{` + prompt + `,` + tc.fields + `,"stream":` + stream + `}`
				status, raw := request(t, ts.Client(), ts.URL+api.path, body, "secret")
				if status != tc.status {
					t.Errorf("%s %s thinking=%v stream=%s: status %d, want %d: %s",
						api.path, tc.fields, tc.think, stream, status, tc.status, raw)
				}
			}
		}
	}
}

func TestAnthropicThinkingBudgetPrecedence(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	for _, tc := range []struct {
		fields string
		status int
	}{
		{`"thinking":{"type":"enabled","budget_tokens":12}`, http.StatusBadRequest},
		{`"thinking":{"type":"enabled","budget_tokens":12},"reasoning_budget_tokens":0`, http.StatusOK},
		{`"thinking":{"type":"enabled","budget_tokens":12},"thinking_budget_tokens":0`, http.StatusBadRequest},
		{`"thinking":{"type":"disabled","budget_tokens":12}`, http.StatusOK},
	} {
		body := `{"messages":[{"role":"user","content":"Hi"}],` + tc.fields + `}`
		status, raw := request(t, ts.Client(), ts.URL+"/v1/messages", body, "secret")
		if status != tc.status {
			t.Errorf("%s: status %d, want %d: %s", tc.fields, status, tc.status, raw)
		}
	}
}

func TestCountTokensIgnoresThinkingBudgetAlias(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"messages":[{"role":"user","content":"Hi"}],"thinking_budget_tokens":"ignored"}`
	status, raw := request(t, ts.Client(), ts.URL+"/v1/messages/count_tokens", body, "secret")
	if status != http.StatusOK {
		t.Fatalf("status %d: %s", status, raw)
	}
}
