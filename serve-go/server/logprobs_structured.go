package server

import "strings"

// logprobSpan returns only entries on exact token boundaries that spell part.
func logprobSpan(text, part string, entries []logprobEntry) []logprobEntry {
	var joined strings.Builder
	for _, entry := range entries {
		joined.WriteString(entry.Token)
	}
	if joined.String() != text {
		return nil
	}
	start := strings.Index(text, part)
	if start < 0 {
		return nil
	}
	at, first := 0, -1
	for i, entry := range entries {
		if at == start && first < 0 {
			first = i
		}
		if at == start+len(part) && first >= 0 {
			return entries[first:i]
		}
		at += len(entry.Token)
	}
	if first >= 0 && at == start+len(part) {
		return entries[first:]
	}
	return nil
}

// structuredLogprobs retains only entries that spell the returned JSON.
func structuredLogprobs(raw, validated string, entries []logprobEntry) (string, []logprobEntry) {
	if len(entries) == 0 {
		return validated, nil
	}
	if raw == validated {
		return validated, entries
	}
	if span := logprobSpan(raw, validated, entries); span != nil {
		return validated, span
	}
	inner := extractJSONObject(raw)
	if canonical, err := validateJSONObject(inner, "stop", false); err == nil && canonical == validated {
		if span := logprobSpan(raw, inner, entries); span != nil {
			return inner, span
		}
	}
	return validated, nil
}
