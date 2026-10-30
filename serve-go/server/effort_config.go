package server

import "strata/requestnorm"

// Python tests request objects for truth before applying a configured default.
func effortTruthy(v requestnorm.Value) bool {
	switch v.Kind() {
	case requestnorm.Null:
		return false
	case requestnorm.Boolean:
		b, _ := v.Bool()
		return b
	case requestnorm.String:
		s, _ := v.Text()
		return s != ""
	case requestnorm.Number:
		n, _ := v.Float64()
		return n != 0
	case requestnorm.Array:
		return len(v.Items()) != 0
	case requestnorm.Object:
		return len(v.Fields()) != 0
	}
	return false
}
