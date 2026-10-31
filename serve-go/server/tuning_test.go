package server

import (
	"net/http"
	"reflect"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

func TestTuningRequestAndDefaults(t *testing.T) {
	defaults, err := ParseSamplingDefaults([]byte(`{"strata_tune":{"pcie_frac":0.2,"spec_min_p":0.7,"aux_cpus":1,"prefill_pipe":42}}`))
	if err != nil {
		t.Fatal(err)
	}
	wantDefault := engine.Sampling{TunePCIEFrac: floatPtr(.2), TuneSpecMinP: floatPtr(.7), TuneAuxCPUs: boolPointer(true)}
	if !reflect.DeepEqual(defaults, wantDefault) {
		t.Fatalf("config tuning = %#v", defaults)
	}
	tok, _ := syntheticTokenizer(t)
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
		for _, test := range []struct {
			field string
			want  engine.Sampling
		}{
			{"", wantDefault},
			{`,"strata_tune":null`, wantDefault},
			{`,"strata_tune":{}`, engine.Sampling{}},
			{`,"strata_tune":false`, engine.Sampling{}},
			{`,"strata_tune":{"pcie_frac":0,"spec_min_p":1,"prefill_pipe":262144,"aux_cpus":false}`, engine.Sampling{TunePCIEFrac: floatPtr(0), TuneSpecMinP: floatPtr(1), TunePrefillPipe: floatPtr(262144), TuneAuxCPUs: boolPointer(false)}},
			{`,"strata_tune":{"pcie_frac":true,"spec_min_p":1.1,"prefill_pipe":-1,"aux_cpus":"1"}`, engine.Sampling{}},
		} {
			body := `{` + api.prefix + test.field + `}`
			w := responsesRequest(h, "http://127.0.0.1"+api.path, body, "")
			want := test.want
			want.Temperature, want.TopP, want.TopK, want.PresencePenalty = floatPtr(0.7), floatPtr(0.8), intPtr(20), floatPtr(1.5)
			if w.Code != http.StatusOK || !reflect.DeepEqual(eng.sampling, want) {
				t.Fatalf("%s %s status=%d got=%#v want=%#v body=%s", api.path, test.field, w.Code, eng.sampling, want, w.Body)
			}
		}
	}
}
