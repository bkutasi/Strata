package server

import (
	"reflect"
	"strconv"
	"strings"
	"testing"

	"strata/requestnorm"
)

func TestParseLogitBias(t *testing.T) {
	tests := []struct {
		name, input, wantErr string
		vocab                int
		want                 map[int]float64
	}{
		{"null", `null`, "", 0, map[int]float64{}},
		{"empty object", `{}`, "", 0, map[int]float64{}},
		{"empty array", `[]`, "", 0, map[int]float64{}},
		{"object", `{"01":-100,"2":100,"3":1.25}`, "", 4, map[int]float64{1: -100, 2: 100, 3: 1.25}},
		{"pairs", `[[0,false],["02",-1e-3],[-0,10]]`, "duplicate token ID", 3, nil},
		{"pair values", `[[0,false],["02",-1e-3],[-0,10]]`, "duplicate token ID", 4, nil},
		{"valid pairs", `[[0,false],["02",-1e-3],[3,100]]`, "", 4, map[int]float64{0: -100, 2: -0.001, 3: 100}},
		{"invalid root", `true`, "expected an object", 4, nil},
		{"invalid pair", `[1]`, "each entry", 4, nil},
		{"short pair", `[[1]]`, "each entry", 4, nil},
		{"long pair", `[[1,2,3]]`, "each entry", 4, nil},
		{"boolean id", `[[false,1]]`, "token IDs", 4, nil},
		{"float id", `[[1.0,1]]`, "token IDs", 4, nil},
		{"negative id", `[[-1,1]]`, "token IDs", 4, nil},
		{"empty id", `{"":1}`, "token IDs", 4, nil},
		{"signed id", `{"+1":1}`, "token IDs", 4, nil},
		{"unicode id", `{"١":1}`, "token IDs", 4, nil},
		{"outside vocab", `{"4":1}`, "outside the model vocabulary", 4, nil},
		{"huge id", `{"999999999999999999999999":1}`, "outside the model vocabulary", 4, nil},
		{"duplicate object", `{"1":0,"01":1}`, "duplicate token ID", 4, nil},
		{"duplicate pair", `[[1,0],["01",1]]`, "duplicate token ID", 4, nil},
		{"false object", `{"1":false}`, "biases must be finite", 4, nil},
		{"true pair", `[[1,true]]`, "biases must be finite", 4, nil},
		{"string bias", `{"1":"1"}`, "biases must be finite", 4, nil},
		{"null bias", `{"1":null}`, "biases must be finite", 4, nil},
		{"below range", `{"1":-100.001}`, "biases must be finite", 4, nil},
		{"above range", `{"1":101}`, "biases must be finite", 4, nil},
		{"huge integer bias", `{"1":` + strings.Repeat("9", 400) + `}`, "biases must be finite", 4, nil},
		{"all banned", `[[0,false],[1,-100]]`, "cannot ban the entire vocabulary", 2, nil},
		{"not all banned", `{"0":-100,"1":-99}`, "", 2, map[int]float64{0: -100, 1: -99}},
		{"zero vocab", `{"0":1}`, "outside the model vocabulary", 0, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			value, err := requestnorm.ParseValue([]byte(tt.input))
			if err != nil {
				t.Fatal(err)
			}
			got, err := parseLogitBias(value, tt.vocab)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("got %v, want error containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, %v; want %v", got, err, tt.want)
			}
		})
	}
}

func TestParseLogitBiasIntegerBoundaries(t *testing.T) {
	// An int32-sized vocabulary can reject MaxInt32 on either architecture.
	for _, input := range []string{`[[2147483647,1]]`, `[[2147483648,1]]`, `[[9223372036854775807,1]]`, `[[9223372036854775808,1]]`} {
		value, err := requestnorm.ParseValue([]byte(input))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := parseLogitBias(value, 2147483647); err == nil {
			t.Errorf("%s: expected error", input)
		}
	}
	if strconv.IntSize == 64 {
		vocab64 := int64(2147483648)
		value, err := requestnorm.ParseValue([]byte(`{"0002147483647":1}`))
		if err != nil {
			t.Fatal(err)
		}
		got, err := parseLogitBias(value, int(vocab64))
		if err != nil || got[2147483647] != 1 {
			t.Fatalf("maximum int32 ID: got %v, %v", got, err)
		}
	}
}
