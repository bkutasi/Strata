package server

import (
	"reflect"
	"testing"
)

func TestExpandImageTokens(t *testing.T) {
	start, pad := 100, 101
	for _, tc := range []struct {
		name    string
		ids     []int
		counts  []int
		limit   int
		want    []int
		wantErr bool
	}{
		{"two images and literal", []int{101, 100, 101, 3, 100, 101}, []int{2, 3}, 10, []int{1, 2, 100, 101, 101, 3, 100, 101, 101, 101}, false},
		{"missing image", []int{100, 101}, nil, 10, nil, true},
		{"extra image", []int{1}, []int{1}, 10, nil, true},
		{"invalid count", []int{100, 101}, []int{0}, 10, nil, true},
		{"too long", []int{100, 101}, []int{10}, 10, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expandImageTokens(tc.ids, start, pad, []int{1, 2}, tc.counts, tc.limit)
			if (err != nil) != tc.wantErr || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, %v; want %v, error=%v", got, err, tc.want, tc.wantErr)
			}
		})
	}
}
