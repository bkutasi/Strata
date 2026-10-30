package server

import (
	"net/http"
	"slices"
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
		s := &server{cfg: Config{Family: tc.family}}
		got := s.effectiveReasoningBudget(tc.thinking, tc.max, tc.request)
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

func TestConfiguredReasoningBudget(t *testing.T) {
	zero, twenty := 0, 20
	for _, tc := range []struct {
		family   requestnorm.Family
		config   *int
		disabled bool
		request  int
		max      int
		want     int
	}{
		{requestnorm.FamilyQwen, &twenty, false, -1, 300, 20},
		{requestnorm.FamilyQwen, &twenty, false, 10, 300, 10},
		{requestnorm.FamilyQwen, &twenty, false, 0, 300, 0},
		{requestnorm.FamilyQwen, &twenty, true, -1, 300, 20},
		{requestnorm.FamilyQwen, nil, true, -1, 3000, 0},
		{requestnorm.FamilyDeepSeek, &zero, true, -1, 300, 0},
		{requestnorm.FamilyDeepSeek, nil, true, -1, 300, 150},
		{requestnorm.FamilyDeepSeek, &zero, false, -1, 300, 150},
	} {
		s := &server{cfg: Config{Family: tc.family, DefaultReasoningBudget: tc.config, DisableAnswerRoom: tc.disabled}}
		if got := s.effectiveReasoningBudget(true, tc.max, tc.request); got != tc.want {
			t.Errorf("%+v: got %d, want %d", tc, got, tc.want)
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

func TestConfiguredReasoningBudgetHTTP(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	budget := 3
	for _, api := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":200}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":200}`},
		{"/v1/responses", `{"input":"hi","max_output_tokens":200}`},
	} {
		eng := &budgetEngine{segments: [][]int{bytesAsTokens("thought"), {'z'}}}
		handler, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
			Engine: eng, Renderer: &toolChoiceRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
			MaxOutputTokens: 2048, MaxConcurrent: 1, DefaultReasoningBudget: &budget, DisableAnswerRoom: true})
		if err != nil {
			t.Fatal(err)
		}
		w := responsesRequest(handler, "http://127.0.0.1"+api.path, api.body, "")
		if w.Code != http.StatusOK || len(eng.prompts) != 2 ||
			len(eng.prompts[1]) < len(eng.prompts[0])+3 ||
			!slices.Equal(eng.prompts[1][len(eng.prompts[0]):len(eng.prompts[0])+3], []int{'t', 'h', 'o'}) {
			t.Errorf("%s: status %d, prompts %v: %s", api.path, w.Code, eng.prompts, w.Body)
		}
	}
}
