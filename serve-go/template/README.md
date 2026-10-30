# Experimental Go chat template renderer

This standalone, opt-in module renders Jinja source with
`github.com/jryberg/gojinja` v1.1.1 (BSD 3-Clause, copyright Johan Ryberg).
It requires Go 1.26; the pinned dependency requests toolchain go1.26.10.
It does not change or connect to Strata's Python serving path.

Pass normalized messages, tools and template kwargs as JSON to `Render`.
`requestnorm` values may be marshaled to JSON for this interface. JSON object
order and integer versus float types are retained. `KeepEmptyTurns` corresponds
to `STRATA_KEEP_EMPTY_TURNS=1`; `AddGenerationPrompt` must be supplied
explicitly. The adapter implements the no-options `tojson` form used in the
two checked-in templates; options fail with an error. `raise_exception` returns
a `RequestError`. Custom templates and other inputs are not established as
Python-compatible.

**Do not wire this into serving without completing the safety boundary.**
Unlike Python's `ImmutableSandboxedEnvironment`, upstream gojinja allows
container mutation. Context cancellation is cooperative and does not cap
allocations; pathological templates can exhaust memory or trigger runtime
failures. A recovered Go panic is returned as an error, but not every runtime
failure is recoverable. The renderer does not mark literal control tokens or
return plain spans: callers need the equivalent of Python's
`mark_think_literals` / `unmark_think_literals` before tokenization, or quoted
special tokens can become real control tokens. No claim of full parity or
safe execution of untrusted templates is made.

`go test ./...` uses the project `.venv` with Jinja2 to normalize the six
DeepSeek fixture inputs and checks all 16 checked-in golden outputs. The tests
do not load model weights or use a GPU.
