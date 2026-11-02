package server

import (
	"net/http"
	"testing"

	"strata/requestnorm"
)

func TestProjectionDefaultsAndRequestPrecedence(t *testing.T) {
	for _, value := range []bool{false, true} {
		data := `{"experimental_speed_projection":false}`
		if value {
			data = `{"experimental_speed_projection":true}`
		}
		defaults, err := ParseSamplingDefaults([]byte(data))
		if err != nil || defaults.SpeedProjection == nil || *defaults.SpeedProjection != value {
			t.Fatalf("default %s: %#v %v", data, defaults, err)
		}
		tok := sharedTokenizer(t)
		eng := &biasCapture{}
		h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
			Engine: eng, Renderer: contextRenderer{}, DefaultSampling: defaults,
			MaxBody: 8192, MaxPromptBytes: 2048, MaxOutputTokens: 128, MaxConcurrent: 1})
		if err != nil {
			t.Fatal(err)
		}
		for _, api := range []struct{ path, prefix string }{
			{"/v1/chat/completions", `"messages":[{"role":"user","content":"hi"}],"max_tokens":32`},
			{"/v1/messages", `"messages":[{"role":"user","content":"hi"}],"max_tokens":32`},
			{"/v1/responses", `"input":"hi","max_output_tokens":32`},
		} {
			for _, option := range []struct {
				field string
				want  *bool
			}{
				{"", &value}, {`,"experimental_speed_projection":null`, &value},
				{`,"experimental_speed_projection":false`, boolPointer(false)},
				{`,"experimental_speed_projection":true`, boolPointer(true)},
				{`,"experimental_speed_projection":0`, nil},
				{`,"experimental_speed_projection":"false"`, nil},
			} {
				body := `{` + api.prefix + option.field + `}`
				w := responsesRequest(h, "http://127.0.0.1"+api.path, body, "")
				got := eng.sampling.SpeedProjection
				if w.Code != http.StatusOK || (got == nil) != (option.want == nil) || (got != nil && *got != *option.want) {
					t.Fatalf("%s %s: status=%d got=%v want=%v body=%s", api.path, option.field, w.Code, got, option.want, w.Body)
				}
			}
		}
		before := eng.calls
		w := responsesRequest(h, "http://127.0.0.1/v1/messages/count_tokens", `{"messages":[{"role":"user","content":"hi"}],"experimental_speed_projection":false}`, "")
		if w.Code != http.StatusOK || eng.calls != before {
			t.Fatalf("count: status=%d calls=%d body=%s", w.Code, eng.calls-before, w.Body)
		}
	}
}

func boolPointer(value bool) *bool { return &value }
