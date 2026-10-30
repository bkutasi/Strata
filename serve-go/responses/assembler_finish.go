package responses

// Finish closes the response and emits its terminal event.
func (a *Assembler) Finish(done Done) ([]Event, error) {
	if err := a.ready(); err != nil {
		return nil, err
	}
	length := done.Finish == "length"
	unfinished := a.item != nil && (a.item["type"] == "function_call" || a.item["type"] == "custom_tool_call")
	status := "completed"
	if length || unfinished {
		status = "incomplete"
	}
	out := a.close(status)
	if a.jsonMode {
		text := a.jsonText.String()
		if a.called {
			released, err := a.release(text)
			if err != nil {
				return out, err
			}
			out = append(out, released...)
		} else {
			validated, err := a.validate(text, done.Finish)
			if err != nil {
				a.terminal = true
				a.validationFailed = true
				return out, err
			}
			released, err := a.release(validated)
			if err != nil {
				return out, err
			}
			out = append(out, released...)
		}
		a.jsonText.Reset()
	}
	prompt := a.prompt
	if done.PromptTokens != nil {
		prompt = *done.PromptTokens
	}
	reused := done.Reused
	if reused < 0 {
		reused = 0
	}
	if reused > prompt {
		reused = prompt
	}
	a.response["usage"] = map[string]any{
		"input_tokens": prompt, "input_tokens_details": map[string]any{"cached_tokens": reused},
		"output_tokens": done.CompletionTokens, "output_tokens_details": map[string]any{"reasoning_tokens": done.ReasoningTokens},
		"total_tokens": prompt + done.CompletionTokens,
	}
	kind := "response.completed"
	if length {
		a.response["status"] = "incomplete"
		a.response["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
		kind = "response.incomplete"
	} else {
		a.response["status"] = "completed"
		a.response["completed_at"] = a.now().Unix()
	}
	a.terminal = true
	final := a.event(kind, map[string]any{"response": a.response})
	if len(done.Timings) > 0 {
		final["timings"] = clone(done.Timings)
	}
	return append(out, final), nil
}

// Failed closes any open item and emits a terminal failure event.
func (a *Assembler) Failed(message, code string) (Event, error) {
	if !a.validationFailed {
		if err := a.ready(); err != nil {
			return nil, err
		}
	}
	a.validationFailed = false
	if code == "" {
		code = "server_error"
	}
	if a.item != nil {
		a.item["status"] = "incomplete"
		a.item = nil
	}
	a.response["status"] = "failed"
	a.response["error"] = map[string]any{"code": code, "message": message}
	a.terminal = true
	return a.event("response.failed", map[string]any{"response": a.response}), nil
}
