package server

import (
	"errors"
	"math"
	"strconv"
	"strings"

	"strata/requestnorm"
)

// -1 means unspecified; zero means the request disabled the explicit budget.
// Values above the host integer range are capped to available output tokens.
func reasoningBudget(root, value requestnorm.Value) (int, error) {
	if value.Kind() == requestnorm.Null {
		value, _ = root.Lookup("thinking_budget_tokens")
	}
	if value.Kind() == requestnorm.Null {
		return -1, nil
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

func effectiveReasoningBudget(family requestnorm.Family, thinking bool, max, budget int) int {
	if !thinking {
		return 0
	}
	if budget < 0 {
		budget = 0
		if family == requestnorm.FamilyDeepSeek {
			budget = 32768
		}
	}
	reserve := min(maxInt(512, max/4), max/2)
	if budget == 0 && max >= 1024 {
		budget = max - reserve
	}
	if budget == 0 && family == requestnorm.FamilyDeepSeek && max >= 48 && max < 1024 {
		budget = max / 2
	}
	if budget > 0 {
		budget = min(budget, maxInt(1, max-reserve))
	}
	return budget
}
