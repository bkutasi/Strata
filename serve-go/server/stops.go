package server

import (
	"errors"

	"strata/requestnorm"
)

func parseStops(root requestnorm.Value) ([]string, error) {
	stop, _ := root.Lookup("stop")
	name := "stop"
	if stop.Kind() == requestnorm.Null {
		stop, _ = root.Lookup("stop_sequences")
		name = "stop_sequences"
	}
	if stop.Kind() == requestnorm.Null {
		return nil, nil
	}
	invalid := name + " must be a list of strings"
	if name == "stop" {
		invalid = "stop must be a string or a list of strings"
	}
	var values []requestnorm.Value
	if name == "stop" && stop.Kind() == requestnorm.String {
		values = []requestnorm.Value{stop}
	} else if stop.Kind() == requestnorm.Array {
		values = stop.Items()
	} else {
		return nil, errors.New(invalid)
	}
	if name == "stop" && len(values) > 4 {
		return nil, errors.New("stop takes at most 4 strings")
	}
	var stops []string
	for _, value := range values {
		text, ok := value.Text()
		if !ok || value.Kind() != requestnorm.String {
			return nil, errors.New(invalid)
		}
		if text != "" {
			stops = append(stops, text)
		}
	}
	return stops, nil
}
