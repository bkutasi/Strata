package server

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"testing"

	"strata/engine"
	"strata/requestnorm"
)

type timedEngine struct{}

func (timedEngine) Alive() bool       { return true }
func (timedEngine) ContextLimit() int { return 4096 }
func (timedEngine) GenerateWithOptions(_ context.Context, _ []int, _ int, opts engine.RequestOptions) (engine.Result, error) {
	if err := opts.OnToken('A'); err != nil {
		return engine.Result{}, err
	}
	return engine.Result{Finish: "stop", Generated: 2, PromptMS: 40, DecodeMS: 40, Extra: []string{"1", "2", "1"}}, nil
}

func TestTimingZeroAndRounding(t *testing.T) {
	timings, err := requestTimings(engine.Result{}, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"prompt_per_second", "prompt_per_token_ms", "predicted_per_second", "predicted_per_token_ms"} {
		if timings[key] != nil {
			t.Errorf("%s = %v, want null", key, timings[key])
		}
	}
	if _, ok := timings["draft_n"]; ok {
		t.Error("old engine unexpectedly has draft counts")
	}
	for _, tc := range []struct {
		input  float64
		digits int
		want   float64
	}{{2.675, 2, 2.67}, {1.25, 1, 1.2}, {1.35, 1, 1.4}} {
		got, err := decimalRound(tc.input, tc.digits)
		if err != nil || got != tc.want {
			t.Errorf("round(%v, %d) = %v, %v; want %v", tc.input, tc.digits, got, err, tc.want)
		}
	}
	if _, err := requestTimings(engine.Result{PromptMS: math.Inf(1)}, 1, 1, 0); err == nil {
		t.Error("non-finite timing accepted")
	}
	if _, err := requestTimings(engine.Result{Extra: []string{"bad", "1", "0"}}, 1, 1, 0); err == nil {
		t.Error("invalid draft count accepted")
	}
}

func TestChatAndResponsesTimings(t *testing.T) {
	tok, _ := syntheticTokenizer(t)
	h, err := New(Config{Model: "test", Family: requestnorm.FamilyQwen, Tokenizer: tok,
		Engine: timedEngine{}, Renderer: contextRenderer{}, MaxBody: 8192,
		MaxPromptBytes: 2048, MaxOutputTokens: 512, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/responses", `{"input":"hi"}`},
	} {
		for _, stream := range []bool{false, true} {
			body := tc.body
			if stream {
				body = strings.TrimSuffix(body, "}") + `,"stream":true}`
			}
			w := responsesRequest(h, "http://127.0.0.1"+tc.path, body, "")
			if w.Code != http.StatusOK {
				t.Fatalf("%s: status %d: %s", tc.path, w.Code, w.Body.String())
			}
			var raw []byte
			if stream {
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if strings.HasPrefix(line, "data: ") && strings.Contains(line, `"timings"`) {
						raw = []byte(strings.TrimPrefix(line, "data: "))
					}
				}
			} else {
				raw = w.Body.Bytes()
			}
			var response struct {
				Timings struct {
					Cache              int     `json:"cache_n"`
					Prompt             int     `json:"prompt_n"`
					PromptPerSecond    float64 `json:"prompt_per_second"`
					Predicted          int     `json:"predicted_n"`
					PredictedPerSecond float64 `json:"predicted_per_second"`
					Draft              int     `json:"draft_n"`
					Accepted           int     `json:"draft_n_accepted"`
				} `json:"timings"`
			}
			if err := json.Unmarshal(raw, &response); err != nil || response.Timings.Cache != 1 ||
				response.Timings.Prompt != 1 || response.Timings.PromptPerSecond != 25 ||
				response.Timings.Predicted != 1 || response.Timings.PredictedPerSecond != 50 ||
				response.Timings.Draft != 2 || response.Timings.Accepted != 1 {
				t.Errorf("%s stream=%v: timings %+v, error %v: %s", tc.path, stream, response.Timings, err, raw)
			}
		}
	}
}
