# Request normalization

This standalone Go module ports the **OpenAI Chat Completions prompt
normalization subset** of `serve/frontend.py` and a separate, limited subset of
Anthropic Messages normalization. It does not start a server, render a
template, load a model, fetch images, or implement the Responses API. It is not
wired into Strata's launch or serving paths.

Use `DecodeOpenAI` to decode the supported JSON fields into `OpenAIRequest`,
then call `NormalizeOpenAI` on a `Normalizer`. Programs may instead construct
the typed request directly. Use `ParseValue` to create tool objects, schemas,
arguments, and effort values; its `Value` retains the insertion order of nested
objects. Call `RequireMessages` separately when an empty conversation is
invalid: the Python conversion and validation are separate too.

```go
n, err := requestnorm.NewNormalizer(requestnorm.FamilyQwen)
if err != nil {
    return err
}
req, err := requestnorm.DecodeOpenAI(requestJSON)
if err != nil {
    return err
}
result, err := n.NormalizeOpenAI(req)
if err != nil {
    return err
}
if err := requestnorm.RequireMessages(result.Messages); err != nil {
    return err
}
// Use result.Messages, result.Tools and result.Kwargs with a compatible template.
```

The conversion handles text, refusals as text, image URLs and image sources,
developer and late system roles, reasoning content and effort, calls, results,
and tools with ordered schemas. It accepts the reference's double-encoded
message, tool-call, and tool lists. It reorders only complete, unambiguous,
consecutive tool result runs to match the previous assistant's call IDs;
otherwise it preserves their received order. IDs remain in `Message.ToolCallID`
and `ToolCall.ID`, even after reordering. The JSON representation of those
objects intentionally excludes the IDs, matching the Python template input.
Keep the IDs separately if later response handling needs them.

Reuse one normalizer for all requests in a server. Its synchronized 64-entry
least-recently-used tool-name-set cache retains the first observed order for a
set of distinct names, as Python does. Duplicate names and single tools are not
cached. Cache state is per normalizer, not a process-global variable; make
requests in a defined order if their first tool order must be deterministic.
Specify `FamilyQwen` or `FamilyDeepSeek` when constructing it. DeepSeek accepts
integer reasoning efforts from 1 through 100.

## Explicit boundaries

`DecodeOpenAI` is strict. It accepts **only** `messages`, `tools`,
`reasoning_effort`, `reasoning.effort`, and the `enable_thinking` and
`reasoning_effort` entries of `chat_template_kwargs`. Any other request field
(including `model`, sampling or streaming settings, `tool_choice`,
`response_format`, and MCP settings) produces `ErrUnsupported`; callers must
handle those features before integration, not discard them. The same rule
applies to unrecognized message, call, content-part, image-source and wrapped
tool fields. Bare tool metadata and arbitrary JSON schema members are preserved
in order. Unsupported part types such as audio, files, or image detail options
are rejected, including system images that Python currently leaves out.
Unsupported roles, non-string IDs and call names, and malformed or lossy
content types are also rejected. This is intentionally stricter than Python,
which sometimes ignores such input. The opt-in `ENV_TO_USER` system-prompt
rewrite is not implemented.

Python's `tool_choice_of` and `forced_call` are separate from
`openai_to_messages`; neither is implemented here. The server's shared
configuration, structured-output handling, local-image policy, literal-token
marking, empty-assistant filtering, rendering, and request validation beyond
the subset above are also outside this package. Consequently it is not a
drop-in replacement for a complete HTTP handler.

`Value` preserves object order and distinguishes arbitrary-precision integers
from IEEE 754 binary64 floats, just as Python's `json.loads` does. `Integer()`
returns the exact integer in decimal (including `-0` normalized to `0`).
`Float64()` returns the rounded float, preserving signed zero and underflow.
`Text()` retains the original spelling, which may describe a *different*
mathematical value: for example, `9007199254740993.0` rounds to
`9007199254740992.0`, and `1e-999` becomes positive zero. Do not render numeric
`Text()` directly. `MarshalJSON` emits the parsed numeric values and retains
their integer/float distinction and the object order, but does **not** implement
Python's `json.dumps` float formatting or spacing.

A future template renderer must traverse `Fields()` in order, use `Integer()`
and `Float64()` to preserve Python's numeric types, and serialize them exactly
as the reference template does with `json.dumps(..., ensure_ascii=False)`.
This includes Python's float representation, the `.0` for integer-valued
floats, signed zero, exponent notation, and default separators and string
escaping. Converting a `Value` into a Go map or relying on `encoding/json` is
not sufficient for byte-identical prompts. Numeric differential tests compare
integer values, float bits, type and object order against the real Python
frontend; they do **not** verify a Go renderer or its output bytes.

Integers above Python's default 4300-digit limit,
float overflow, duplicate object keys, invalid UTF-8, lone surrogate escapes,
and Python's non-standard `NaN`/`Infinity` JSON constants are rejected rather
than coerced. A non-object, non-string tool argument containing a float is
rejected because Python stringifies that fallback using its own float
formatting; malformed argument *strings* otherwise retain the reference's
`{"arguments": "original text"}` fallback. Reasoning values other than the
documented strings, false, or a DeepSeek integer and non-boolean
`enable_thinking` values are rejected. Python accepts or handles some of these
cases differently.

Run `go test -race ./...`, `go vet ./...`, and `go build ./...` here. The
differential test runs the actual `serve.frontend` with the project
`../../.venv/bin/python`, with no model, GPU, server, or downloaded assets. If
that interpreter is unavailable, only the differential test is skipped.
