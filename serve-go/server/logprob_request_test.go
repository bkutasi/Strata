package server

import (
	"reflect"
	"testing"

	"strata/requestnorm"
)

func TestChatLogprobsRequest(t *testing.T) {
	for _, test := range []struct {
		json    string
		count   *int
		allowed []int
		bad     bool
	}{
		{`{}`, nil, nil, false},
		{`{"logprobs":false,"allowed_token_ids":[999]}`, nil, nil, false},
		{`{"logprobs":true}`, intPtr(0), nil, false},
		{`{"logprobs":true,"top_logprobs":-3}`, intPtr(0), nil, false},
		{`{"logprobs":true,"top_logprobs":3}`, intPtr(3), nil, false},
		{`{"logprobs":true,"top_logprobs":999999999999999999999999}`, intPtr(20), nil, false},
		{`{"logprobs":true,"top_logprobs":true}`, intPtr(0), nil, false},
		{`{"logprobs":0}`, intPtr(0), nil, false},
		{`{"logprobs":-0}`, intPtr(0), nil, false},
		{`{"logprobs":99}`, intPtr(20), nil, false},
		{`{"logprobs":-1}`, nil, nil, false},
		{`{"logprobs":0.0}`, nil, nil, false},
		{`{"logprobs":true,"allowed_token_ids":[1,true,-1,2.0,2,"3"]}`, intPtr(0), []int{1, 2}, false},
		{`{"logprobs":true,"allowed_token_ids":[3]}`, nil, nil, true},
		{`{"logprobs":true,"allowed_token_ids":[999999999999999999999999]}`, nil, nil, true},
	} {
		root, err := requestnorm.ParseValue([]byte(test.json))
		if err != nil {
			t.Fatal(err)
		}
		count, allowed, err := chatLogprobs(root, 3)
		if (err != nil) != test.bad || !reflect.DeepEqual(count, test.count) || !reflect.DeepEqual(allowed, test.allowed) {
			t.Errorf("%s: count=%v allowed=%v error=%v", test.json, count, allowed, err)
		}
	}
}
