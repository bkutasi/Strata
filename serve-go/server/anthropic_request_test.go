package server

import (
	"strings"
	"testing"

	"strata/requestnorm"
)

func anthropicDecoderServer() *server {
	n, _ := requestnorm.NewNormalizer(requestnorm.FamilyQwen)
	return &server{cfg: Config{Model: "model", MaxOutputTokens: 128}, normalizer: n}
}

func TestDecodeAnthropicRequest(t *testing.T) {
	s := anthropicDecoderServer()
	tests := []struct {
		name, body string
		stream     bool
		wantErr    string
		stops      string
		topK       int
		noTopK     bool
	}{
		{"happy", `{"model":"model","max_tokens":20,"messages":[{"role":"user","content":"hi"}],"temperature":0.7}`, false, "", "", 0, false},
		{"stream", `{"model":"model","max_tokens":20,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, true, "", "", 0, false},
		{"invalid sampling", `{"model":"model","messages":[{"role":"user","content":"hi"}],"temperature":[]}`, false, "temperature", "", 0, false},
		{"tool choice", `{"model":"model","messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"auto"}}`, false, "tool_choice", "", 0, false},
		{"budget", `{"model":"model","messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":12}`, false, "reasoning budget", "", 0, false},
		{"stop precedence", `{"model":"model","messages":[{"role":"user","content":"hi"}],"stop":["a"],"stop_sequences":["b"]}`, false, "", "a", 0, false},
		{"many stop sequences", `{"model":"model","messages":[{"role":"user","content":"hi"}],"stop_sequences":["a","b","c","d","e"]}`, false, "", "", 0, false},
		{"invalid stop sequences", `{"model":"model","messages":[{"role":"user","content":"hi"}],"stop_sequences":"halt"}`, false, "stop_sequences", "", 0, false},
		{"null stop sequences", `{"model":"model","messages":[{"role":"user","content":"hi"}],"stop_sequences":null}`, false, "", "", 0, false},
		{"thinking budget", `{"model":"model","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":12}}`, false, "reasoning budget", "", 0, false},
		{"top_k integer", `{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":12}`, false, "", "", 12, false},
		{"top_k string", `{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":"12"}`, false, "", "", 12, false},
		{"top_k clamp", `{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":100}`, false, "", "", 64, false},
		{"top_k negative", `{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":-2}`, false, "", "", 0, true},
		{"top_k null", `{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":null}`, false, "", "", 0, true},
		{"top_k fractional", `{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":1.5}`, false, "top_k", "", 0, false},
		{"empty", `{"model":"model","messages":[]}`, false, "messages", "", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, opt, err := s.decodeAnthropicRequest([]byte(tt.body))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if opt.stream != tt.stream {
				t.Fatalf("stream = %v, want %v", opt.stream, tt.stream)
			}
			if tt.stops != "" && (len(opt.stops) != 1 || opt.stops[0] != tt.stops) {
				t.Fatalf("stops = %v, want [%s]", opt.stops, tt.stops)
			}
			if tt.name == "many stop sequences" && len(opt.stops) != 5 {
				t.Fatalf("stop sequences = %v, want 5 entries", opt.stops)
			}
			if tt.noTopK && opt.sampling.TopK != nil {
				t.Fatalf("top_k = %d, want unset", *opt.sampling.TopK)
			}
			if tt.topK > 0 && (opt.sampling.TopK == nil || *opt.sampling.TopK != tt.topK) {
				t.Fatalf("top_k = %v, want %d", opt.sampling.TopK, tt.topK)
			}
			if len(got.Messages) == 0 {
				t.Fatal("normalized messages are empty")
			}
		})
	}
}
