package server

import (
	"encoding/json"
	"math"
	"testing"
)

func TestParseSamplingDefaultsValid(t *testing.T) {
	tests := []struct {
		name string
		data string
		want map[string]any
	}{
		{"null block", `null`, map[string]any{}},
		{"null values and off values", `{"temperature":0,"top_p":1,"min_p":0,"repetition_penalty":1,"frequency_penalty":0,"presence_penalty":0,"penalty_last_n":0}`, map[string]any{}},
		{"positive defaults and aliases", `{"temperature":0.7,"top_p":0.9,"top_k":12.0,"min_p":0.1,"repeat_penalty":1.2,"frequency_penalty":0.2,"presence_penalty":0.3,"repeat_last_n":128.0,"seed":42}`, map[string]any{
			"temperature": float64(0.7), "top_p": float64(0.9), "top_k": int(12), "min_p": float64(0.1),
			"repetition_penalty": float64(1.2), "frequency_penalty": float64(0.2), "presence_penalty": float64(0.3),
			"penalty_last_n": int(128), "seed": uint64(42),
		}},
		{"null block values", `{"temperature":null,"top_k":null,"seed":null}`, map[string]any{}},
		{"alias replaces null canonical", `{"repetition_penalty":null,"repeat_penalty":1.2,"penalty_last_n":null,"repeat_last_n":12}`, map[string]any{"repetition_penalty": float64(1.2), "penalty_last_n": int(12)}},
		{"unknown values ignored", `{"unknown":{"nested":true}}`, map[string]any{}},
		{"float seed rounds like Python", `{"seed":9007199254740993.0}`, map[string]any{"seed": uint64(9007199254740992)}},
		{"uint64 maximum", `{"seed":18446744073709551615}`, map[string]any{"seed": uint64(math.MaxUint64)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ParseSamplingDefaults([]byte(test.data))
			if err != nil {
				t.Fatal(err)
			}
			actual := map[string]any{}
			if got.Temperature != nil {
				actual["temperature"] = *got.Temperature
			}
			if got.TopP != nil {
				actual["top_p"] = *got.TopP
			}
			if got.TopK != nil {
				actual["top_k"] = *got.TopK
			}
			if got.MinP != nil {
				actual["min_p"] = *got.MinP
			}
			if got.RepetitionPenalty != nil {
				actual["repetition_penalty"] = *got.RepetitionPenalty
			}
			if got.FrequencyPenalty != nil {
				actual["frequency_penalty"] = *got.FrequencyPenalty
			}
			if got.PresencePenalty != nil {
				actual["presence_penalty"] = *got.PresencePenalty
			}
			if got.PenaltyLastN != nil {
				actual["penalty_last_n"] = *got.PenaltyLastN
			}
			if got.Seed != nil {
				actual["seed"] = *got.Seed
			}
			if len(actual) != len(test.want) {
				t.Fatalf("got %#v, want %#v", actual, test.want)
			}
			for key, want := range test.want {
				a, _ := json.Marshal(actual[key])
				w, _ := json.Marshal(want)
				if string(a) != string(w) {
					t.Errorf("%s: got %s, want %s", key, a, w)
				}
			}
		})
	}
}

func TestParseSamplingDefaultsInvalid(t *testing.T) {
	tests := []string{
		`[]`, `"not an object"`, `{"temperature":"0.7"}`, `{"temperature":true}`,
		`{"temperature":-1}`, `{"temperature":3.4028236e38}`,
		`{"top_p":0}`, `{"top_p":1.1}`, `{"top_p":1e-1000}`,
		`{"min_p":-0.1}`, `{"min_p":1.1}`,
		`{"top_k":0}`, `{"top_k":65}`, `{"top_k":1.5}`, `{"top_k":1e100}`,
		`{"repeat_penalty":1.2,"repetition_penalty":1.1}`,
		`{"repeat_last_n":2,"penalty_last_n":3}`,
		`{"repeat_penalty":0}`, `{"repetition_penalty":3.4028236e38}`,
		`{"frequency_penalty":-1}`, `{"frequency_penalty":3.4028236e38}`,
		`{"presence_penalty":-1}`, `{"penalty_last_n":-1}`,
		`{"penalty_last_n":2147483648}`, `{"penalty_last_n":2.5}`,
		`{"seed":0}`, `{"seed":-1}`, `{"seed":1.5}`, `{"seed":18446744073709551616}`,
		`{"strata_tune":"bad"}`, `{"experimental_speed_projection":0}`, `{"experimental_speed_projection":"false"}`,
	}
	for _, data := range tests {
		t.Run(data, func(t *testing.T) {
			if _, err := ParseSamplingDefaults([]byte(data)); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}
