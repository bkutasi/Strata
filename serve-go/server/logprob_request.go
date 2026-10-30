package server

import (
	"errors"
	"strconv"
	"strings"

	"strata/requestnorm"
)

// chatLogprobs follows Python's request rules. A nil count means no logprobs.
func chatLogprobs(root requestnorm.Value, vocabularySize int) (*int, []int, error) {
	value, _ := root.Lookup("logprobs")
	var count int
	if enabled, ok := value.Bool(); ok && enabled {
		top, _ := root.Lookup("top_logprobs")
		if integer, ok := top.Integer(); ok {
			if strings.HasPrefix(integer, "-") {
				count = 0
			} else if n, err := strconv.Atoi(integer); err != nil || n > 20 {
				count = 20
			} else {
				count = n
			}
		}
	} else if integer, ok := value.Integer(); ok && !strings.HasPrefix(integer, "-") {
		if n, err := strconv.Atoi(integer); err != nil || n > 20 {
			count = 20
		} else {
			count = n
		}
	} else {
		return nil, nil, nil
	}
	allowedValue, _ := root.Lookup("allowed_token_ids")
	var allowed []int
	if allowedValue.Kind() == requestnorm.Array {
		for _, value := range allowedValue.Items() {
			integer, ok := value.Integer()
			if !ok || strings.HasPrefix(integer, "-") {
				continue
			}
			id, err := strconv.Atoi(integer)
			if err != nil || id >= vocabularySize {
				return nil, nil, errors.New("allowed_token_ids contains an ID outside the model vocabulary")
			}
			allowed = append(allowed, id)
		}
	}
	return &count, allowed, nil
}
