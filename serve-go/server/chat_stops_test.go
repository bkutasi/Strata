package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestChatStopSequences(t *testing.T) {
	for _, tc := range []struct {
		fields, want string
	}{
		{`"stop":null`, "before STOP hidden"},
		{`"stop":null,"stop_sequences":["STOP"]`, "before "},
		{`"stop_sequences":["x","y","z","none","STOP"]`, "before "},
		{`"stop":"hidden","stop_sequences":"ignored"`, "before STOP "},
		{`"stop":[],"stop_sequences":["STOP"]`, "before STOP hidden"},
		{`"stop":["","STOP"]`, "before "},
		{`"stop":"` + strings.Repeat("a", 1025) + `"`, "before STOP hidden"},
	} {
		for _, stream := range []bool{false, true} {
			h := responsesHandler(t, "before STOP hidden", "stop", false)
			body := `{"messages":[{"role":"user","content":"hi"}],` + tc.fields
			if stream {
				body += `,"stream":true`
			}
			w := responsesRequest(h, "/v1/chat/completions", body+`}`, "secret")
			if w.Code != http.StatusOK {
				t.Fatalf("%s stream=%v: %d %s", tc.fields, stream, w.Code, w.Body.String())
			}
			var text string
			if stream {
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if !strings.HasPrefix(line, "data: {") {
						continue
					}
					var chunk struct {
						Choices []struct {
							Delta struct {
								Content string `json:"content"`
							} `json:"delta"`
						} `json:"choices"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
						t.Fatal(err)
					}
					if len(chunk.Choices) != 0 {
						text += chunk.Choices[0].Delta.Content
					}
				}
			} else {
				var response struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || len(response.Choices) != 1 {
					t.Fatalf("%s: %v", w.Body.String(), err)
				}
				text = response.Choices[0].Message.Content
			}
			if text != tc.want {
				t.Errorf("%s stream=%v: got %q, want %q", tc.fields, stream, text, tc.want)
			}
		}
	}
}

func TestChatInvalidStops(t *testing.T) {
	h := responsesHandler(t, "hello", "stop", false)
	for _, fields := range []string{
		`"stop":42`, `"stop":[1]`, `"stop":["a","b","c","d","e"]`,
		`"stop_sequences":"STOP"`, `"stop":null,"stop_sequences":[null]`,
	} {
		body := `{"messages":[{"role":"user","content":"hi"}],` + fields + `}`
		w := responsesRequest(h, "/v1/chat/completions", body, "secret")
		if w.Code != http.StatusBadRequest {
			t.Errorf("%s: %d %s", fields, w.Code, w.Body.String())
		}
	}
}
