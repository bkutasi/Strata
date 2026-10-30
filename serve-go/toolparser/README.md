# Tool-call body parser

This module ports the body parser and framing primitives, not `OutputParser`.
Python JSON integers are represented as `json.Number`, and JSON floating-point
values as `float64`. This preserves arbitrary-size integers and Python's
binary64 rounding, signed zero, infinities and NaN. The integer digit limit
matches CPython's default of 4300 digits, not a runtime-customized limit.

Python's server uses `json.dumps` for tool arguments (including in OpenAI
stream chunks and Anthropic input deltas). Its default `allow_nan=True` emits
`NaN` and `Infinity`, which are not valid JSON numbers. This module does not
provide an HTTP serializer: Go's `encoding/json` rejects non-finite floats,
including inside arrays and objects. A downstream server must explicitly
choose whether to reject such calls or transform their values before producing
standards-compliant JSON; silently converting them here would lose parser
parity. OpenAI argument strings are also JSON documents even when their outer
HTTP envelope is valid.

Go strings are UTF-8, whereas Python strings can contain lone surrogate code
points. `ParseToolCall` returns `ErrUnsupportedUnicode` for invalid UTF-8 input
or valid JSON values containing unpaired surrogate escapes, including object
keys and values later overwritten by duplicate keys. Python accepts those
JSON values. Valid surrogate pairs are supported. Invalid JSON remains the
original parameter text, and schema-declared string values are not JSON-decoded.
This is an explicit compatibility gap rather than a lossy replacement.

`go test ./...` runs standalone unit tests without model assets. When the
project `.venv` is present it also runs the Python differential oracle;
otherwise the oracle test is skipped.
