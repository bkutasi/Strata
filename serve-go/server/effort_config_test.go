package server

import (
	"testing"

	"strata/requestnorm"
	"strata/responses"
)

func TestConfiguredReasoningEffort(t *testing.T) {
	n, err := requestnorm.NewNormalizer(requestnorm.FamilyQwen)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: Config{Model: "test", Family: requestnorm.FamilyQwen, MaxOutputTokens: 512, DefaultReasoningEffort: "low"}, normalizer: n}
	for _, tc := range []struct {
		name, fields string
		want         string
	}{
		{"default", "", `{"reasoning_effort":"low"}`},
		{"explicit", `,"reasoning_effort":"none"`, `{"enable_thinking":false}`},
		{"template override", `,"chat_template_kwargs":{"enable_thinking":false}`, `{"enable_thinking":false}`},
		{"fallback", `,"reasoning":{"effort":"high"}`, `{"reasoning_effort":"xhigh"}`},
	} {
		t.Run("chat/"+tc.name, func(t *testing.T) {
			got, _, err := s.decode([]byte(`{"messages":[{"role":"user","content":"hi"}]` + tc.fields + `}`))
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := got.Kwargs.MarshalJSON()
			if string(raw) != tc.want {
				t.Fatalf("kwargs %s, want %s", raw, tc.want)
			}
		})
	}
	for _, tc := range []struct {
		name, fields string
		want         string
	}{
		{"default", "", `{"reasoning_effort":"low"}`},
		{"explicit disabled", `,"thinking":{"type":"disabled"}`, `{"enable_thinking":false}`},
		{"explicit high", `,"output_config":{"effort":"high"}`, `{"reasoning_effort":"xhigh"}`},
	} {
		t.Run("anthropic/"+tc.name, func(t *testing.T) {
			req := []byte(`{"messages":[{"role":"user","content":"hi"}]` + tc.fields + `}`)
			got, err := s.decodeAnthropicCount(req)
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := got.Kwargs.MarshalJSON()
			if string(raw) != tc.want {
				t.Fatalf("kwargs %s, want %s", raw, tc.want)
			}
		})
	}
	t.Run("responses", func(t *testing.T) {
		req, err := responses.Decode([]byte(`{"input":"hi"}`))
		if err != nil {
			t.Fatal(err)
		}
		effort, _ := requestnorm.ParseValue([]byte(`"low"`))
		got, err := responses.Prepare(req, responses.PrepareOptions{Family: requestnorm.FamilyQwen, SharedReasoningEffort: effort})
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := got.Kwargs.MarshalJSON()
		if string(raw) != `{"reasoning_effort":"low"}` {
			t.Fatalf("kwargs %s", raw)
		}
	})
}
