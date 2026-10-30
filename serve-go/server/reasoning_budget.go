package server

import (
	"errors"
	"math"
	"strconv"
	"strings"

	"strata/requestnorm"
)

// A non-positive budget disables the limit. Values above the host integer
// range are capped later to the request's available output tokens.
func reasoningBudget(root, value requestnorm.Value) (int, error) {
	if value.Kind() == requestnorm.Null {
		value, _ = root.Lookup("thinking_budget_tokens")
	}
	if value.Kind() == requestnorm.Null {
		return 0, nil
	}
	if integer, ok := value.Integer(); ok {
		if integer == "0" || strings.HasPrefix(integer, "-") {
			return 0, nil
		}
		n, err := strconv.Atoi(integer)
		if err != nil {
			return math.MaxInt, nil
		}
		return n, nil
	}
	if number, ok := value.Float64(); ok && math.Trunc(number) == number {
		if number <= 0 {
			return 0, nil
		}
		if number >= float64(math.MaxInt) {
			return math.MaxInt, nil
		}
		return int(number), nil
	}
	return 0, errors.New("reasoning_budget_tokens must be a whole number")
}
