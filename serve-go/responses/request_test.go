package responses

import (
	"encoding/json"
	"testing"

	"strata/requestnorm"
)

func TestDecodeAndPrepareTextInput(t *testing.T) {
	req, err := Decode([]byte(`{"instructions":"Be brief.","input":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := Prepare(req, PrepareOptions{Family: requestnorm.FamilyQwen})
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(prepared.Messages)
	if err != nil {
		t.Fatal(err)
	}
	const want = `[{"role":"system","content":"Be brief."},{"role":"user","content":"hi"}]`
	if string(got) != want {
		t.Fatalf("messages = %s, want %s", got, want)
	}
}

func TestDecodeAndPrepareFunctionCallInput(t *testing.T) {
	req, err := Decode([]byte(`{"input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"shell.exec","arguments":"{\"cmd\":\"pwd\"}"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := Prepare(req, PrepareOptions{Family: requestnorm.FamilyQwen})
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(prepared.Messages)
	if err != nil {
		t.Fatal(err)
	}
	const want = `[{"role":"user","content":"run"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"shell.exec","arguments":{"cmd":"pwd"}}}]}]`
	if string(got) != want {
		t.Fatalf("messages = %s, want %s", got, want)
	}
	if call := prepared.Messages[1].ToolCalls[0]; call.ID != "c1" || call.Function.Name != "shell.exec" {
		t.Fatalf("call mapping = %#v", call)
	}
}

func TestPrepareRejectsStatefulRequest(t *testing.T) {
	req, err := Decode([]byte(`{"input":"hi","previous_response_id":"resp_1"}`))
	if err != nil {
		t.Fatal(err)
	}
	_, err = Prepare(req, PrepareOptions{Family: requestnorm.FamilyQwen})
	if err == nil {
		t.Fatal("Prepare accepted previous_response_id")
	}
	apiErr, ok := err.(*Error)
	if !ok {
		t.Fatalf("error type = %T, want *Error", err)
	}
	if apiErr.Param != "previous_response_id" || apiErr.Code != "unsupported_parameter" || apiErr.Status != 400 {
		t.Fatalf("error fields = %#v", apiErr)
	}
}
