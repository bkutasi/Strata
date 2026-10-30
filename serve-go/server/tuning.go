package server

import (
	"math"
	"strconv"

	"strata/engine"
	"strata/requestnorm"
)

// parseTuning keeps the native keys Python sends; config excludes prefill_pipe.
func parseTuning(value requestnorm.Value, config bool) engine.Sampling {
	var out engine.Sampling
	if value.Kind() != requestnorm.Object {
		return out
	}
	for _, field := range value.Fields() {
		if field.Name == "aux_cpus" {
			if b, ok := field.Value.Bool(); ok {
				out.TuneAuxCPUs = &b
			} else if field.Value.Kind() == requestnorm.Number {
				text, _ := field.Value.Text()
				n, err := strconv.ParseFloat(text, 64)
				if err == nil && (n == 0 || n == 1) {
					b := n == 1
					out.TuneAuxCPUs = &b
				}
			}
			continue
		}
		max := 1.0
		switch field.Name {
		case "pcie_frac", "spec_min_p":
		case "prefill_pipe":
			if config {
				continue
			}
			max = 262144
		default:
			continue
		}
		if field.Value.Kind() != requestnorm.Number {
			continue
		}
		text, _ := field.Value.Text()
		n, err := strconv.ParseFloat(text, 64)
		if err != nil || math.IsNaN(n) || math.IsInf(n, 0) || n < 0 || n > max {
			continue
		}
		switch field.Name {
		case "pcie_frac":
			out.TunePCIEFrac = &n
		case "spec_min_p":
			out.TuneSpecMinP = &n
		case "prefill_pipe":
			out.TunePrefillPipe = &n
		}
	}
	return out
}

func setTuning(s *engine.Sampling, tune engine.Sampling) {
	s.TunePCIEFrac = tune.TunePCIEFrac
	s.TuneSpecMinP = tune.TuneSpecMinP
	s.TunePrefillPipe = tune.TunePrefillPipe
	s.TuneAuxCPUs = tune.TuneAuxCPUs
}
