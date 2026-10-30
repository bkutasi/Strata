package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUnknownRequestedModelUsesLoadedModel(t *testing.T) {
	h, _ := setup(t)
	ts := httptest.NewServer(h)
	defer ts.Close()
	for _, api := range []struct {
		path, prompt string
	}{
		{"/v1/chat/completions", `"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"enable_thinking":false}`},
		{"/v1/messages", `"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"disabled"}`},
		{"/v1/responses", `"input":"Hi","reasoning":{"effort":"none"}`},
	} {
		for _, model := range []string{`"other"`, `null`, `123`} {
			body := `{` + api.prompt + `,"model":` + model + `,"max_tokens":32}`
			if api.path == "/v1/responses" {
				body = `{` + api.prompt + `,"model":` + model + `,"max_output_tokens":32}`
			}
			status, raw := request(t, ts.Client(), ts.URL+api.path, body, "secret")
			var response struct {
				Model string `json:"model"`
			}
			err := json.Unmarshal([]byte(raw), &response)
			if status != http.StatusOK || err != nil || response.Model != "test" {
				t.Errorf("%s model=%s: status %d, response %s, error %v", api.path, model, status, raw, err)
			}
		}
		body := `{` + api.prompt + `,"model":"other","stream":true}`
		status, raw := request(t, ts.Client(), ts.URL+api.path, body, "secret")
		modelFound := false
		scanner := bufio.NewScanner(strings.NewReader(raw))
		for scanner.Scan() {
			data, ok := strings.CutPrefix(scanner.Text(), "data: ")
			if !ok || data == "[DONE]" {
				continue
			}
			var event map[string]any
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				t.Fatalf("%s: invalid stream event: %v", api.path, err)
			}
			if event["model"] == "test" {
				modelFound = true
			}
			for _, key := range []string{"message", "response"} {
				if nested, ok := event[key].(map[string]any); ok && nested["model"] == "test" {
					modelFound = true
				}
			}
		}
		if err := scanner.Err(); err != nil {
			t.Fatal(err)
		}
		if status != http.StatusOK || !modelFound {
			t.Errorf("%s stream: status %d, response %s", api.path, status, raw)
		}
	}
}
