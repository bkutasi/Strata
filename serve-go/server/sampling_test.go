package server

import (
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

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
				return s.PenaltyLastN == nil && s.Temperature == nil && s.TopP == nil && s.MinP == nil && s.Seed == nil
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
