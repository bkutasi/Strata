package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/big"
	"strconv"

	"strata/engine"
	"strata/requestnorm"
)

// ParseSamplingDefaults parses the supported sampling defaults from a config
// block. A JSON null block means that no defaults are configured.
func ParseSamplingDefaults(data []byte) (engine.Sampling, error) {
	var out engine.Sampling
	data = bytes.TrimSpace(data)
	if bytes.Equal(data, []byte("null")) {
		return out, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return out, fmt.Errorf("sampling defaults must be an object or null: %w", err)
	}
	if fields == nil {
		return out, errors.New("sampling defaults must be an object or null")
	}

	for _, pair := range [][2]string{{"repeat_penalty", "repetition_penalty"}, {"repeat_last_n", "penalty_last_n"}} {
		alias, canonical := fields[pair[0]], fields[pair[1]]
		if alias == nil || isJSONNull(alias) {
			continue
		}
		if canonical != nil && !isJSONNull(canonical) {
			return out, fmt.Errorf("sampling: both %q and %q are set; they are the same setting", pair[0], pair[1])
		}
		fields[pair[1]] = alias
		delete(fields, pair[0])
	}

	for key, raw := range fields {
		if isJSONNull(raw) {
			continue
		}
		if key == "strata_tune" {
			value, err := requestnorm.ParseValue(raw)
			if err != nil || value.Kind() != requestnorm.Object {
				return out, errors.New("sampling.strata_tune: expected an object")
			}
			setTuning(&out, parseTuning(value, true))
			continue
		}
		if key == "experimental_speed_projection" {
			var value bool
			if err := json.Unmarshal(raw, &value); err != nil {
				return out, fmt.Errorf("sampling.%s: expected true or false", key)
			}
			out.SpeedProjection = &value
			continue
		}
		switch key {
		case "temperature", "top_p", "min_p", "top_k", "repetition_penalty", "frequency_penalty",
			"presence_penalty", "penalty_last_n", "seed":
		default:
			log.Printf("[strata] config sampling.%s: unknown key, ignored", key)
			continue
		}
		value, err := parseSamplingNumber(raw)
		if err != nil {
			return out, fmt.Errorf("sampling.%s: expected a JSON number: %w", key, err)
		}
		switch key {
		case "temperature":
			if value < 0 || value > math.MaxFloat32 {
				return out, errors.New("sampling.temperature: expected a number >= 0 representable by the engine")
			}
			if value != 0 {
				out.Temperature = floatPtr(value)
			}
		case "top_p":
			if value <= 0 || value > 1 || value < math.SmallestNonzeroFloat32 {
				return out, errors.New("sampling.top_p: expected 0 < top_p <= 1 representable by the engine")
			}
			if value != 1 {
				out.TopP = floatPtr(value)
			}
		case "min_p":
			if value < 0 || value > 1 {
				return out, errors.New("sampling.min_p: expected 0 <= min_p <= 1")
			}
			if value != 0 {
				out.MinP = floatPtr(value)
			}
		case "top_k":
			n, err := parseConfigInteger(raw, value)
			if err != nil || !n.IsInt64() || n.Int64() < 1 || n.Int64() > 64 {
				return out, errors.New("sampling.top_k: expected an integer 1..64")
			}
			out.TopK = intPtr(int(n.Int64()))
		case "repetition_penalty":
			if value <= 0 || value > math.MaxFloat32 || value < math.SmallestNonzeroFloat32 {
				return out, errors.New("sampling.repetition_penalty: expected a positive number representable by the engine")
			}
			if value != 1 {
				out.RepetitionPenalty = floatPtr(value)
			}
		case "frequency_penalty":
			if value < 0 || value > math.MaxFloat32 {
				return out, errors.New("sampling.frequency_penalty: expected a number >= 0 representable by the engine")
			}
			if value != 0 {
				out.FrequencyPenalty = floatPtr(value)
			}
		case "presence_penalty":
			if value < 0 || value > math.MaxFloat32 {
				return out, errors.New("sampling.presence_penalty: expected a number >= 0 representable by the engine")
			}
			if value != 0 {
				out.PresencePenalty = floatPtr(value)
			}
		case "penalty_last_n":
			n, err := parseConfigInteger(raw, value)
			if err != nil || !n.IsInt64() || n.Int64() < 0 || n.Int64() > math.MaxInt32 {
				return out, errors.New("sampling.penalty_last_n: expected a non-negative engine-representable integer")
			}
			if n.Int64() != 0 {
				out.PenaltyLastN = intPtr(int(n.Int64()))
			}
		case "seed":
			n, err := parseConfigInteger(raw, value)
			if err != nil || n.Sign() <= 0 || !n.IsUint64() {
				return out, errors.New("sampling.seed: expected a positive uint64 integer")
			}
			out.Seed = uint64Ptr(n.Uint64())
		}
	}
	return out, nil
}

func isJSONNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

func parseSamplingNumber(raw json.RawMessage) (float64, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || (raw[0] != '-' && (raw[0] < '0' || raw[0] > '9')) {
		return 0, errors.New("not a JSON number")
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err != nil {
		return 0, err
	}
	return strconv.ParseFloat(number.String(), 64)
}

func parseConfigInteger(raw json.RawMessage, value float64) (*big.Int, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) || math.Trunc(value) != value {
		return nil, errors.New("not integral")
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err != nil {
		return nil, err
	}
	n, ok := new(big.Int).SetString(number.String(), 10)
	if !ok {
		// Python parses decimal and exponent forms as float before int().
		n, _ = new(big.Float).SetFloat64(value).Int(nil)
	}
	return n, nil
}

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int) *int           { return &v }
func uint64Ptr(v uint64) *uint64  { return &v }
