package server

import (
	"errors"
	"math"
	"strconv"

	"strata/requestnorm"
)

// parseLogitBias validates request-scoped token biases before generation.
func parseLogitBias(value requestnorm.Value, vocabSize int) (map[int]float64, error) {
	out := make(map[int]float64)
	bans := 0
	add := func(token string, bias requestnorm.Value, pair bool) error {
		if token == "" {
			return errors.New("logit_bias: token IDs must be non-negative integers")
		}
		var id int64
		overflow := false
		for i := 0; i < len(token); i++ {
			if token[i] < '0' || token[i] > '9' {
				return errors.New("logit_bias: token IDs must be non-negative integers")
			}
			if !overflow {
				id = id*10 + int64(token[i]-'0')
				overflow = id > math.MaxInt32
			}
		}
		if overflow || id >= int64(vocabSize) {
			return errors.New("logit_bias: token ID outside the model vocabulary")
		}
		if _, exists := out[int(id)]; exists {
			return errors.New("logit_bias: duplicate token ID")
		}
		var n float64
		var valid bool
		if b, ok := bias.Bool(); ok && !b && pair {
			n, valid = -100, true
		} else if f, ok := bias.Float64(); ok {
			n, valid = f, true
		} else if s, ok := bias.Integer(); ok {
			var err error
			n, err = strconv.ParseFloat(s, 64)
			valid = err == nil
		}
		if !valid || math.IsNaN(n) || math.IsInf(n, 0) || n < -100 || n > 100 {
			return errors.New("logit_bias: biases must be finite numbers between -100 and 100; false is a pair-list ban")
		}
		out[int(id)] = n
		if n == -100 {
			bans++
		}
		return nil
	}

	switch value.Kind() {
	case requestnorm.Null:
		return out, nil
	case requestnorm.Object:
		for _, field := range value.Fields() {
			if err := add(field.Name, field.Value, false); err != nil {
				return nil, err
			}
		}
	case requestnorm.Array:
		for _, entry := range value.Items() {
			if entry.Kind() != requestnorm.Array {
				return nil, errors.New("logit_bias: each entry must be [token_id, bias]")
			}
			items := entry.Items()
			if len(items) != 2 {
				return nil, errors.New("logit_bias: each entry must be [token_id, bias]")
			}
			var token string
			var ok bool
			if items[0].Kind() == requestnorm.String {
				token, ok = items[0].Text()
			} else {
				token, ok = items[0].Integer()
			}
			if !ok {
				return nil, errors.New("logit_bias: token IDs must be non-negative integers")
			}
			if err := add(token, items[1], true); err != nil {
				return nil, err
			}
		}
	default:
		return nil, errors.New("logit_bias: expected an object or a list of [token_id, bias] pairs")
	}
	if len(out) > 0 && bans == vocabSize {
		return nil, errors.New("logit_bias: cannot ban the entire vocabulary")
	}
	return out, nil
}
