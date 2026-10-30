package server

import (
	"net/http"
	"testing"

	"strata/requestnorm"
)

func TestAutomaticReasoningBudget(t *testing.T) {
	for _, tc := range []struct {
		family   requestnorm.Family
		thinking bool
		max      int
		request  int
		want     int
	}{
		{requestnorm.FamilyQwen, true, 1023, -1, 0},
		{requestnorm.FamilyQwen, true, 1024, -1, 512},
		{requestnorm.FamilyQwen, true, 3000, 0, 2250},
		{requestnorm.FamilyQwen, true, 300, 1000, 150},
		{requestnorm.FamilyQwen, false, 3000, 20, 0},
		{requestnorm.FamilyDeepSeek, true, 47, 0, 0},
		{requestnorm.FamilyDeepSeek, true, 48, 0, 24},
		{requestnorm.FamilyDeepSeek, true, 300, 0, 150},
		{requestnorm.FamilyDeepSeek, true, 300, -1, 150},
		{requestnorm.FamilyDeepSeek, true, 100000, -1, 32768},
	} {
		got := effectiveReasoningBudget(tc.family, tc.thinking, tc.max, tc.request)
		if got != tc.want {
			t.Errorf("%+v: got %d, want %d", tc, got, tc.want)
		}
	}
}

func TestReasoningBudgetDistinguishesMissingAndDisabled(t *testing.T) {
	for _, tc := range []struct {
		body string
		want int
	}{
		{`{}`, -1},
		{`{"reasoning_budget_tokens":null}`, -1},
		{`{"reasoning_budget_tokens":0}`, 0},
		{`{"reasoning_budget_tokens":-10}`, 0},
		{`{"reasoning_budget_tokens":null,"thinking_budget_tokens":0}`, 0},
	} {
		root, err := requestnorm.ParseValue([]byte(tc.body))
		if err != nil {
			t.Fatal(err)
		}
		value, _ := root.Lookup("reasoning_budget_tokens")
		got, err := reasoningBudget(root, value)
		if err != nil || got != tc.want {
			t.Errorf("%s: budget %d, error %v, want %d", tc.body, got, err, tc.want)
		}
	}
}

func TestAutomaticReasoningBudgetHTTP(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	thought := make([]int, 600)
	for i := range thought {
		thought[i] = 'a'
	}
	for _, api := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":1024}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":1024}`},
		{"/v1/responses", `{"input":"hi","max_output_tokens":1024}`},
	} {
		eng := &budgetEngine{segments: [][]int{thought, {'z'}}}
		handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
			Engine: eng, Renderer: &toolChoiceRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
			MaxOutputTokens: 2048, MaxConcurrent: 1})
		if err != nil {
			t.Fatal(err)
		}
		w := responsesRequest(handler, "http://127.0.0.1"+api.path, api.body, "")
		if w.Code != http.StatusOK || len(eng.prompts) != 2 {
			t.Errorf("%s: status %d, prompts %d: %s", api.path, w.Code, len(eng.prompts), w.Body)
		}
	}
}
