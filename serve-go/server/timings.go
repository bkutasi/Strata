package server

import (
	"fmt"
	"math"
	"strconv"

	"strata/engine"
)

// decimalRound rounds the actual float value, as Python's round does, rather
// than rounding an intermediate multiplication.
func decimalRound(value float64, digits int) (float64, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("non-finite engine timing")
	}
	return strconv.ParseFloat(strconv.FormatFloat(value, 'f', digits, 64), 64)
}

func requestTimings(result engine.Result, prompt, generated, cached int) (map[string]any, error) {
	promptRead := prompt - cached
	decoded := result.Generated
	if decoded == 0 {
		decoded = generated
	}
	rounded := func(value float64, places int) (any, error) {
		return decimalRound(value, places)
	}
	promptMS, err := rounded(result.PromptMS, 1)
	if err != nil {
		return nil, err
	}
	decodeMS, err := rounded(result.DecodeMS, 1)
	if err != nil {
		return nil, err
	}
	timings := map[string]any{
		"cache_n": cached, "prompt_n": promptRead, "prompt_ms": promptMS,
		"prompt_per_token_ms": nil, "prompt_per_second": nil,
		"predicted_n": generated, "predicted_ms": decodeMS,
		"predicted_per_token_ms": nil, "predicted_per_second": nil,
	}
	if promptRead > 0 {
		if timings["prompt_per_token_ms"], err = rounded(result.PromptMS/float64(promptRead), 3); err != nil {
			return nil, err
		}
		if result.PromptMS > 0 {
			if timings["prompt_per_second"], err = rounded(float64(promptRead)/(result.PromptMS/1000), 1); err != nil {
				return nil, err
			}
		}
	}
	if decoded > 0 {
		if timings["predicted_per_token_ms"], err = rounded(result.DecodeMS/float64(decoded), 3); err != nil {
			return nil, err
		}
		if result.DecodeMS > 0 {
			if timings["predicted_per_second"], err = rounded(float64(decoded)/(result.DecodeMS/1000), 1); err != nil {
				return nil, err
			}
		}
	}
	if len(result.Extra) >= 3 {
		accepted, err := strconv.Atoi(result.Extra[0])
		if err != nil || accepted < 0 {
			return nil, fmt.Errorf("invalid engine accepted draft count %q", result.Extra[0])
		}
		offered, err := strconv.Atoi(result.Extra[1])
		if err != nil || offered < 0 {
			return nil, fmt.Errorf("invalid engine offered draft count %q", result.Extra[1])
		}
		timings["draft_n"], timings["draft_n_accepted"] = offered, accepted
	}
	return timings, nil
}
