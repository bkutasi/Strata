# Incremental output parser

This is a standalone Go component for the Qwen-style `OutputParser` in
`serve/frontend.py`. It does not replace or change the Python server. It has
its own module and imports the adjacent `strata/toolparser` with a local
`replace`; no model, engine, network, or GPU is needed.

## Use

Create one parser per generation with `New(Options{...})`. `Thinking` must be
set explicitly: unlike Python's constructor, the Go zero value is false.
Pass normalized tools as `[]map[string]any`, using the same `name` and
`parameters.properties` shape as the Python frontend. Treat the tool schemas
and their nested maps as immutable for that parser's lifetime. Schema
`parameters`, `properties`, and individual property values must be objects
or null; invalid shapes are rejected at construction.

Call `Feed` once for each valid UTF-8 text delta, in order, and process its
events in the order returned. Finish once with `Finish("")` for Python's
`None`, `"stop"` for a natural stop, or the actual reason such as `"length"`.
`Status` gives the state, clean-boundary flag, pending reasoning-call count,
and unread/rescued/refused counters. A clean buffer can still have pending
reasoning calls; check both when a continuation requires no held calls.
A parser is single-use and must not be accessed concurrently.

Each new call receives a `call_` ID with the same 24-hex-digit UUID-v4 prefix
shape as Python. Set `Options.ID` to a unique, deterministic ID generator to
replay a recorded sequence of feeds and obtain the same IDs and event
boundaries. The caller owns the recording, stop handling, transport encoding,
and resource limits; this module does not implement any of them.

`tool_start` and `tool_args` are provisional progress for display or transport,
not permission to run a tool. Execute only a final `tool_call` event; some
announced calls never finish, and their streamed fragments can differ from the
final parsed arguments. Process a returned error before acting on any events in
that same batch. Python-compatible recovery at a natural stop after
`</function>` without `</tool_call>` can emit a final streamed call with empty
arguments, even if fragments were sent earlier. Validate the final arguments
before execution.

An error is terminal. A `Feed` or `Finish` can return events *and* an error:
those events were produced before the error, in order. Abort the generation
and its client stream on any error; do not execute a call from that batch,
present it as a successful answer, or resume the parser. In particular, a
streamed call may already have been announced when its unsupported argument
is detected.

## Supported

- Incremental reasoning and content, split tags, quoted `</think>` markers,
  newline handling, Markdown fence/inline-code tracking, and per-feed event
  order.
- Wrapped native tool calls, schema-based argument decoding, namespace aliases
  and exact-name precedence, unknown-tool reporting, literal tags in argument
  values, consecutive calls, and the native unclosed-wrapper boundary rule.
- Calls inside reasoning held until they can be distinguished from quotation;
  natural stop versus cut-off behavior and corresponding counters.
- `stream_tools` start/argument events, including incremental declared string
  parameters, and final calls with the same ID. For other parameter types,
  streaming supports strings, null, booleans, arbitrary-size integers, arrays
  of supported values, and objects with at most one key, recursively.
- Valid Unicode text and the numeric representations of the adjacent
  `toolparser`: JSON integers are `json.Number` and floats are `float64`.

## Explicit compatibility gaps

- `Recover: true` is rejected by `New`. Bare function calls, JSON or parameter
  wrappers, and batches within one wrapper that require Python's opt-in
  recovery mode are not implemented. Native loose openers that work without
  recovery are implemented.
- With `StreamTools`, a parsed non-string parameter containing a float,
  including non-finite floats, or a multi-key JSON object at any depth returns
  `ErrUnsupported`. Python's exact float rendering and object insertion order
  cannot be recovered from the current Go tool-parser value representation.
  Without streamed tool arguments these values remain available in the final
  parsed call. A caller must separately decide how to serialize non-finite
  floats; Go's standard JSON encoder rejects them.
- Each feed must itself be valid UTF-8; split a Unicode code point only before
  the detokenizer, not between `Feed` calls. A valid JSON value containing
  unpaired surrogate escapes returns `toolparser.ErrUnsupportedUnicode`
  instead of losing data. The body parser's other documented numeric and
  Unicode limits also apply.
- DSML rewriting and its DeepSeek-specific parser are not implemented.
  `Clean` is exposed through `Status`, not as a synchronization primitive.
  Memory/time use is not bounded by this package: configure output/token
  limits upstream, especially for incomplete calls and very long lines.

An individual call body is buffered with amortized linear copying as chunks
arrive. Structural framing and streamed parameter-end searches retain their
progress across feeds; confirming a final call still parses its complete body.
An incomplete standard first `<function=` opener, its preceding wrapper
whitespace, and whitespace following a pending reasoning call are also
retained incrementally.
The surrounding content and reasoning handling can retain other long text, and
a feed containing many calls can copy their remaining text repeatedly. Loose
opener forms are not covered by the incremental opener buffer and can still
cause repeated copying or scanning. Callers still need limits.

These are limits of this component, not claims that it is ready to replace
the Python serving path. It does not provide an HTTP serializer, stop matcher,
tokenizer, model, or server.

## Verify

Run `go test -race ./...`, `go vet ./...`, and `go build ./...` in this
directory. Differential tests use the project `.venv/bin/python` and read
the existing native-tool and reasoning corpus fixtures. If that Python is
unavailable, Go-only tests still run but differential tests are skipped.
