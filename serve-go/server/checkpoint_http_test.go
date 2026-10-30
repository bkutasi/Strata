package server

import (
	"net/http"
	"testing"

	"strata/requestnorm"
)

func TestCheckpointOptionHTTP(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	eng := &biasCapture{}
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: eng, Renderer: contextRenderer{}, MaxBody: 8192, MaxPromptBytes: 2048,
		MaxOutputTokens: 128, MaxConcurrent: 1})
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
			want  bool
		}{
			{``, false}, {`,"strata_checkpoint":false`, true},
			{`,"strata_checkpoint":true`, false}, {`,"strata_checkpoint":0`, false},
			{`,"strata_checkpoint":null`, false}, {`,"strata_checkpoint":"false"`, false},
		} {
			for _, stream := range []string{"false", "true"} {
				body := `{` + api.prefix + option.field + `,"stream":` + stream + `}`
				w := responsesRequest(h, "http://127.0.0.1"+api.path, body, "")
				if w.Code != http.StatusOK || eng.sampling.DisableCheckpoint != option.want {
					t.Fatalf("%s %s stream=%s: status=%d disabled=%v body=%s", api.path, option.field, stream, w.Code, eng.sampling.DisableCheckpoint, w.Body)
				}
			}
		}
	}
	before := eng.calls
	w := responsesRequest(h, "http://127.0.0.1/v1/messages/count_tokens", `{"messages":[{"role":"user","content":"hi"}],"strata_checkpoint":false}`, "")
	if w.Code != http.StatusOK || eng.calls != before {
		t.Fatalf("count: status=%d calls=%d body=%s", w.Code, eng.calls-before, w.Body)
	}
}
