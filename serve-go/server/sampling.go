package server

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"

	"strata/engine"
	"strata/requestnorm"
)

func samplingFloat(name string, value requestnorm.Value) (*float64, error) {
	if value.Kind() == requestnorm.Null {
		return nil, nil
	}
	if value.Kind() != requestnorm.String && value.Kind() != requestnorm.Number {
		return nil, fmt.Errorf("%s must be a number", name)
	}
	text, _ := value.Text()
	text = strings.TrimSpace(text)
	if value.Kind() == requestnorm.String && strings.HasPrefix(strings.ToLower(strings.TrimLeft(text, "+-")), "0x") {
		return nil, fmt.Errorf("%s must be a number", name)
	}
	n, err := strconv.ParseFloat(text, 64)
	if err != nil || math.IsNaN(n) || math.IsInf(n, 0) {
		return nil, fmt.Errorf("%s must be a finite number", name)
	}
	return &n, nil
}

func samplingInteger(name string, value requestnorm.Value) (*big.Int, error) {
	if value.Kind() == requestnorm.Null {
		return nil, nil
	}
	if text, ok := value.Integer(); ok {
		n, _ := new(big.Int).SetString(text, 10)
		return n, nil
	}
	if value.Kind() == requestnorm.String {
		n, err := samplingFloat(name, value)
		if err == nil && math.Trunc(*n) == *n {
			result, _ := new(big.Float).SetFloat64(*n).Int(nil)
			return result, nil
		}
	}
	return nil, fmt.Errorf("%s must be an integer", name)
}

func setSampling(s *engine.Sampling, root requestnorm.Value, name string, value requestnorm.Value) error {
	if name == "repeat_penalty" {
		if canonical, ok := root.Lookup("repetition_penalty"); ok && canonical.Kind() != requestnorm.Null {
			return nil
		}
		name = "repetition_penalty"
	}
	if name == "repeat_last_n" {
		if canonical, ok := root.Lookup("penalty_last_n"); ok && canonical.Kind() != requestnorm.Null {
			return nil
		}
		// Python only applies this alias when it is a positive JSON integer.
		if _, ok := value.Integer(); !ok {
			return nil
		}
		name = "penalty_last_n"
	}
	if name == "seed" || name == "penalty_last_n" {
		n, err := samplingInteger(name, value)
		if err != nil || n == nil {
			return err
		}
		if n.Sign() <= 0 {
			return nil
		}
		if name == "seed" {
			if !n.IsUint64() {
				return fmt.Errorf("%s is out of range", name)
			}
			seed := n.Uint64()
			s.Seed = &seed
			return nil
		}
		if !n.IsInt64() || n.Int64() > math.MaxInt32 {
			return fmt.Errorf("%s is out of range", name)
		}
		window := int(n.Int64())
		s.PenaltyLastN = &window
		return nil
	}
	n, err := samplingFloat(name, value)
	if err != nil || n == nil {
		return err
	}
	switch name {
	case "temperature":
		if *n <= 0 {
			return nil
		}
		if *n > math.MaxFloat32 {
			return fmt.Errorf("%s is out of range", name)
		}
		s.Temperature = n
	case "top_p":
		if *n <= 0 {
			return fmt.Errorf("%s is out of range", name)
		}
		if *n < 1 {
			s.TopP = n
		}
	case "min_p":
		if *n > 0 && *n <= 1 {
			s.MinP = n
		}
	case "repetition_penalty":
		if *n < math.SmallestNonzeroFloat32 || *n > math.MaxFloat32 {
			return fmt.Errorf("%s is out of range", name)
		}
		s.RepetitionPenalty = n
	case "frequency_penalty", "presence_penalty":
		if math.Abs(*n) > math.MaxFloat32 {
			return fmt.Errorf("%s is out of range", name)
		}
		if name == "frequency_penalty" {
			s.FrequencyPenalty = n
		} else {
			s.PresencePenalty = n
		}
	}
	return nil
}
