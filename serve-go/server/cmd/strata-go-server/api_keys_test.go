package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestConfigAPIKeys(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want []string
		bad  bool
	}{
		{"missing", "", nil, false},
		{"null", "null", nil, false},
		{"empty string", `""`, nil, false},
		{"none", `" NoNe "`, nil, false},
		{"comma separated", `" a, b,, "`, []string{"a", "b"}, false},
		{"list preserves commas and none", `[" a,b ", "none"]`, []string{"a,b", "none"}, false},
		{"blank string", `" , "`, nil, true},
		{"empty list", `[]`, nil, true},
		{"blank list key", `["a", " "]`, nil, true},
		{"non-string list key", `["a", 1]`, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := configAPIKeys(json.RawMessage(tc.raw))
			if (err != nil) != tc.bad || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("keys = %#v, error = %v; want %#v, bad = %t", got, err, tc.want, tc.bad)
			}
		})
	}
}
