package server

import (
	"context"
	"net/http"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type samplingRenderer struct{ thinking bool }

func (r samplingRenderer) Render(context.Context, requestnorm.Result) (Rendered, error) {
	return Rendered{Prompt: "Hi", Thinking: r.thinking}, nil
}

func (r samplingRenderer) RenderAnthropic(context.Context, requestnorm.AnthropicResult) (Rendered, error) {
	return Rendered{Prompt: "Hi", Thinking: r.thinking}, nil
}

func TestQwenModelSamplingAcrossAPIs(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	for _, tc := range []struct {
		name, config, fields string
		thinking             bool
		temperature, topP    float64
		topK                 int
		presence             float64
	}{
		{"thinking", `{}`, ``, true, 1, 0.95, 20, 0},
		{"answering", `{}`, ``, false, 0.7, 0.8, 20, 1.5},
		{"config off", `{"temperature":0,"top_p":1,"top_k":7,"presence_penalty":0}`, ``, false, 0, 1, 7, 0},
		{"config partial", `{"temperature":0.4}`, ``, true, 0.4, 0.95, 20, 0},
		{"request off", `{"temperature":0.4,"top_p":0.5,"top_k":7,"presence_penalty":0.2}`,
			`,"temperature":0,"top_p":1,"top_k":0,"presence_penalty":0`, false, 0, 1, 64, 0},
		{"request partial", `{"temperature":0.4}`, `,"temperature":0.2,"top_p":0.6`, false, 0.2, 0.6, 20, 1.5},
		{"null falls back", `{"temperature":0.4}`, `,"temperature":null,"top_p":null`, false, 0.4, 0.8, 20, 1.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defaults, err := ParseSamplingDefaults([]byte(tc.config))
			if err != nil {
				t.Fatal(err)
			}
			eng := &samplingCapture{}
			h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: eng,
				Renderer: samplingRenderer{thinking: tc.thinking}, APIKeys: []string{"secret"},
				MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1, DefaultSampling: defaults})
			if err != nil {
				t.Fatal(err)
			}
			for _, api := range []struct{ path, prefix string }{
				{"/v1/chat/completions", `"messages":[{"role":"user","content":"hi"}]`},
				{"/v1/messages", `"messages":[{"role":"user","content":"hi"}],"max_tokens":32`},
				{"/v1/responses", `"input":"hi"`},
			} {
				t.Run(api.path, func(t *testing.T) {
					w := responsesRequest(h, api.path, `{`+api.prefix+tc.fields+`}`, "secret")
					eng.mu.Lock()
					got := eng.got
					eng.mu.Unlock()
					temperature, topP, topK, presence := 0.0, 1.0, 20, 0.0
					if got.Temperature != nil {
						temperature = *got.Temperature
					}
					if got.TopP != nil {
						topP = *got.TopP
					}
					if got.TopK != nil {
						topK = *got.TopK
					}
					if got.PresencePenalty != nil {
						presence = *got.PresencePenalty
					}
					if w.Code != http.StatusOK || temperature != tc.temperature || topP != tc.topP ||
						topK != tc.topK || presence != tc.presence {
						t.Fatalf("status %d, effective sampling %v/%v/%v/%v; want %v/%v/%v/%v: %s",
							w.Code, temperature, topP, topK, presence,
							tc.temperature, tc.topP, tc.topK, tc.presence, w.Body.String())
					}
				})
			}
		})
	}
}

func TestSamplingDefaultsAcrossAPIs(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	eng := &samplingCapture{}
	temperature, topP, minP, repetition := 0.7, 0.9, 0.1, 1.1
	frequency, presence, topK, window, seed := 0.2, 0.3, 20, 16, uint64(42)
	defaults := engine.Sampling{Temperature: &temperature, TopP: &topP, TopK: &topK, MinP: &minP,
		RepetitionPenalty: &repetition, FrequencyPenalty: &frequency, PresencePenalty: &presence,
		PenaltyLastN: &window, Seed: &seed}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok, Engine: eng,
		Renderer: contextRenderer{}, APIKeys: []string{"secret"}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 512, MaxConcurrent: 1, DefaultSampling: defaults})
	if err != nil {
		t.Fatal(err)
	}
	for _, api := range []struct{ path, prefix string }{
		{"/v1/chat/completions", `"messages":[{"role":"user","content":"hi"}]`},
		{"/v1/messages", `"messages":[{"role":"user","content":"hi"}],"max_tokens":32`},
		{"/v1/responses", `"input":"hi"`},
	} {
		for _, tc := range []struct {
			name, fields string
			check        func(engine.Sampling) bool
		}{
			{"defaults", "", func(s engine.Sampling) bool {
				return s.Temperature != nil && *s.Temperature == 0.7 && s.TopP != nil && *s.TopP == 0.9 &&
					s.TopK != nil && *s.TopK == 20 && s.MinP != nil && *s.MinP == 0.1 &&
					s.RepetitionPenalty != nil && *s.RepetitionPenalty == 1.1 &&
					s.FrequencyPenalty != nil && *s.FrequencyPenalty == 0.2 &&
					s.PresencePenalty != nil && *s.PresencePenalty == 0.3 &&
					s.PenaltyLastN != nil && *s.PenaltyLastN == 16 && s.Seed != nil && *s.Seed == 42
			}},
			{"null falls back", `,"temperature":null,"top_k":null,"repeat_last_n":0`, func(s engine.Sampling) bool {
				return s.Temperature != nil && *s.Temperature == 0.7 && s.TopK != nil && *s.TopK == 20 &&
					s.PenaltyLastN != nil && *s.PenaltyLastN == 16
			}},
			{"explicit off", `,"temperature":0,"top_p":1,"min_p":0,"seed":0,"penalty_last_n":0`, func(s engine.Sampling) bool {
				return s.Temperature != nil && *s.Temperature == 0 && s.TopP != nil && *s.TopP == 1 &&
					s.MinP == nil && s.Seed == nil && s.PenaltyLastN == nil
			}},
			{"aliases", `,"repeat_penalty":1.4,"repeat_last_n":10`, func(s engine.Sampling) bool {
				return s.RepetitionPenalty != nil && *s.RepetitionPenalty == 1.4 && s.PenaltyLastN != nil && *s.PenaltyLastN == 10
			}},
		} {
			t.Run(api.path+"/"+tc.name, func(t *testing.T) {
				w := responsesRequest(h, api.path, `{`+api.prefix+tc.fields+`}`, "secret")
				eng.mu.Lock()
				got := eng.got
				eng.mu.Unlock()
				if w.Code != http.StatusOK || !tc.check(got) {
					t.Fatalf("status %d, sampling %#v: %s", w.Code, got, w.Body.String())
				}
			})
		}
	}
}

func TestSamplingAliasesAndIntegerBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name, input string
		check       func(engine.Sampling) bool
	}{
		{"canonical wins", `{"repeat_penalty":2,"repetition_penalty":1.2,"repeat_last_n":10,"penalty_last_n":20}`,
			func(s engine.Sampling) bool {
				return s.RepetitionPenalty != nil && *s.RepetitionPenalty == 1.2 && s.PenaltyLastN != nil && *s.PenaltyLastN == 20
			}},
		{"alias after null", `{"repetition_penalty":null,"repeat_penalty":"1.1","penalty_last_n":null,"repeat_last_n":16}`,
			func(s engine.Sampling) bool {
				return s.RepetitionPenalty != nil && *s.RepetitionPenalty == 1.1 && s.PenaltyLastN != nil && *s.PenaltyLastN == 16
			}},
		{"ignored alias string", `{"repeat_last_n":"16","temperature":-1,"top_p":2,"min_p":1.2,"seed":-1}`,
			func(s engine.Sampling) bool {
				return s.PenaltyLastN == nil && s.Temperature != nil && *s.Temperature == 0 &&
					s.TopP != nil && *s.TopP == 1 && s.MinP == nil && s.Seed == nil
			}},
		{"uint64 seed", `{"seed":18446744073709551615}`,
			func(s engine.Sampling) bool { return s.Seed != nil && *s.Seed == ^uint64(0) }},
		{"string rounds like Python", `{"seed":"9007199254740993"}`,
			func(s engine.Sampling) bool { return s.Seed != nil && *s.Seed == 9007199254740992 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, err := requestnorm.ParseValue([]byte(tc.input))
			if err != nil {
				t.Fatal(err)
			}
			var sampling engine.Sampling
			for _, field := range root.Fields() {
				if err := setSampling(&sampling, root, field.Name, field.Value); err != nil {
					t.Fatal(err)
				}
			}
			if !tc.check(sampling) {
				t.Fatalf("unexpected sampling: %#v", sampling)
			}
		})
	}
	for _, input := range []string{`{"seed":18446744073709551616}`, `{"penalty_last_n":2147483648}`} {
		root, err := requestnorm.ParseValue([]byte(input))
		if err != nil {
			t.Fatal(err)
		}
		field := root.Fields()[0]
		if err := setSampling(&engine.Sampling{}, root, field.Name, field.Value); err == nil {
			t.Fatalf("accepted out-of-range input %s", input)
		}
	}
}
