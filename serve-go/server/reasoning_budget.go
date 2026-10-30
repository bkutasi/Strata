package server

import (
	"errors"
	"math"
	"strings"

	"strata/requestnorm"
)

// A non-positive budget disables the limit. Positive budgets are only safe to
// ignore when the rendered prompt has thinking disabled.
func reasoningBudget(root, value requestnorm.Value) (bool, error) {
	if value.Kind() == requestnorm.Null {
		value, _ = root.Lookup("thinking_budget_tokens")
	}
	if value.Kind() == requestnorm.Null {
		return false, nil
	}
	if integer, ok := value.Integer(); ok {
		return integer != "0" && !strings.HasPrefix(integer, "-"), nil
	}
	if number, ok := value.Float64(); ok && math.Trunc(number) == number {
		return number > 0, nil
	}
	return false, errors.New("reasoning_budget_tokens must be a whole number")
}

func rejectActiveReasoningBudget(budget, thinking bool) error {
	if budget && thinking {
		return errors.New("reasoning budget with thinking is not supported")
	}
	return nil
}
