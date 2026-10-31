# Experimental Go server

This is an opt-in experiment, not a replacement for the Python server. It is
not connected to `strata launch` or the desktop app. It accepts Qwen and
DeepSeek V4.1 requests at `POST /v1/chat/completions`, with streaming and a
limited set of request fields, lists the model at `GET /v1/models` and
`GET /models`, and reports
health at `GET /health` and `GET /api/health`. It also accepts
Anthropic requests at `POST /v1/messages`, including streaming, and counts
prompts at `POST /v1/messages/count_tokens` without running generation.
As in Python, requests naming another model use the loaded model. Configure
`aliases` in the standalone JSON config as a list of names or one
comma-separated string. Aliases are trimmed, deduplicated, and listed at
`/v1/models`; requests naming an alias report that alias, while unknown names
report the loaded model name.
It also accepts stateless requests at `POST /v1/responses`, with
named streaming events, reasoning, stop strings, and completed function and
custom tool calls. Hosted and unknown tools are left out and a warning is
logged; the server does not run them. Stored responses are not supported.
As in the Python server, `store: true` is accepted but responses are not stored.
Codex client metadata is accepted; the optional Codex thread-title and
compaction-cache behaviors are not implemented.
The standalone server accepts `--no-responses` and `--no-anthropic` to make
those API routes return 404. These options do not unload statically linked Go
code and do not guarantee lower process RSS.
Non-finite generated tool arguments fail the response. Tool
argument object keys may be reordered by the parser.
The APIs support `auto` and `none` tool choices. Chat and Anthropic requests
also support forced or named choices with or without thinking. A positive request reasoning budget closes
thinking and continues the answer; zero or negative disables the explicit
limit. As in Python, automatic answer-room limits can still apply. The run
config supports `reasoning_budget_tokens` and `answer_room: false`. A reply
that repeats one token 256 times is stopped as length; set
`repeat_stop_tokens` to 0 to turn this off.
Set `anthropic_thinking` to `on_request` to turn thinking off for Anthropic
requests that do not ask for it; the default is `model`.
As in the
Python Responses route, Responses `tool_choice: none` removes tools; other
choices are accepted but do not force a tool. Incomplete tool calls are not
streamed. Image support and renderer limits are described below. Other
endpoints and model families are not implemented.
Chat, Responses and Anthropic sampling supports temperature, top-p, top-k, min-p, repetition,
frequency and presence penalties, penalty window, and seed.
For Qwen, the fallback matches Python's `qwen4exp` defaults: thinking uses temperature
1.0, top-p 0.95 and top-k 20; without thinking it uses temperature 0.7, top-p 0.8,
top-k 20 and presence penalty 1.5. Config and request values override these defaults.
Chat, Responses and Anthropic requests also accept `logit_bias` as an object
or a list of token-ID and bias pairs. It requires a native engine that
advertises bias support and is not using continuous batching; unsupported
engines reject the request.
For one-shot requests, `"strata_checkpoint": false` skips creating a
conversation checkpoint, as in Python; cached prefixes can still be reused.
`experimental_speed_projection` can be set to true or false per request or
in `sampling` defaults. It affects engines started with a control vector.
Chat accepts `logprobs`, `top_logprobs` and `allowed_token_ids`. An older
engine that does not send probabilities returns the answer without them.
As in Python, Chat accepts but ignores `n` and `parallel_tool_calls`; it
returns one choice and does not enforce that tool-call flag.
When Chat supplies both output-token limits, `max_completion_tokens` takes
precedence unless it is zero or null, as in the Python server.
Chat, Responses and Anthropic accept `stop` or `stop_sequences`; a non-null
`stop` takes precedence and is limited to four strings.
Chat and Responses support JSON-object and JSON Schema output, including
optional LaTeX escape repair. Schema output is prompted and validated, not
constrained during generation; remote schema references are refused. Unlike
Python when its optional `jsonschema` package is absent, Go always validates
the schema. Go's regular expression and schema-draft behavior can differ from
Python's validator. Like the Python server, Chat rejects structured output
when tools are enabled; Responses applies it to the final answer rather than
tool calls.

From `serve-go/server`, build with Go 1.26.10:

```sh
go build -o strata-go-server ./cmd/strata-go-server
./strata-go-server -help
```

Run it only with a trusted model pack and template. The template renderer does
not provide Python's immutable sandbox. Quoted control tokens in request values
are encoded as ordinary text; template-generated tokens remain special. As in
Python, assistant tool-call replay and outer user tool-result wrappers retain
their markers. Control tokens in object keys are still rejected.
See `../template/README.md` for the renderer's limits.

This server requires the tokenizer files, template, and a native engine
executable supplied using its command-line options. The engine arguments must
include `--serve` and the model flags. It defaults to listening on loopback;
an API key is required for non-loopback addresses.
Use `--family deepseek41` for DeepSeek. The pack must include a compatible
V4.1 template; unlike Python, this experiment does not replace older templates.

The server also accepts a limited standalone Python JSON run config:
`strata-go-server --config /absolute/path/run.json --port 50000 --api-key-env STRATA_API_KEY`.
For compatibility with the Python command, `--engine strata` can also be
given with `--config`; the executable still comes from the config.
The Go server starts its own engine; it cannot attach to the engine owned by
an already running Python server. It checks the listening port before starting
the engine. Running two engines needs enough memory for both model loads.
It supports `exe`, `cwd`, `args`, `tokenizer`, `model_name`, `host`, `api_key`,
integer `gpu`, string-valued `env`, `log`, `reasoning_budget_tokens`,
`reasoning_effort`, `answer_room`, `fit_max_tokens`, `repeat_stop_tokens`, `anthropic_thinking`
and the `sampling` block for standard sampling parameters. The optional
`reasoning_effort` accepts `none`, `low`, `medium` or `high`; it supplies the
default across Chat, Responses and Anthropic APIs, and an explicit request
value takes precedence. Sampling defaults apply when a request omits the field
or sets it to null; explicit request values take precedence.
Output limits are shortened to fit the context by default. Set
`"fit_max_tokens": false` to reject an explicit positive output limit that
exceeds the remaining context; omitted, zero and negative limits still use
the available context, subject to `MaxOutputTokens`.
`strata_tune` accepts Python's existing per-request engine tuning keys; its
config default accepts `pcie_frac`, `spec_min_p`, and `aux_cpus`.
Other config fields are rejected rather than silently ignored.
`--api-key` overrides the config key, while
`--api-key-env` takes precedence over both. As in Python, `--api-key`,
`STRATA_API_KEY` and a config string split keys on commas. A config list keeps
each key intact, including commas. `--api-key-env` takes its environment
variable as one literal key. The config's host is used unless
`--listen` is supplied; check it before starting a network-accessible server.
This path has not been validated with every Python config.

Qwen image input can use the existing native `strata-vision` reader when the
standalone config has a `vision` object with `exe`, `mmproj`, and `model`
paths. The optional `gpu`, `threads`, `max_tokens`, and `min_tokens` settings
are supported. Relative paths are resolved against the config's `cwd`. The
server starts the reader before the engine and adds `--vision` to the engine
arguments. Chat, Anthropic and Responses accept image inputs as data URLs,
HTTP(S) URLs, or local files; foreign-origin web pages cannot read local
files. The reader and engine take turns, including for text-only requests.
With `--lazy-vision`, the server starts the reader for each image and stops it
afterward instead of keeping it in memory. Each image pays the reader's startup
cost. A GPU reader then needs enough VRAM free after the engine starts; it can
fail if the engine has taken that space. This option has not been tested with
a real vision model.
The server limits image data to 32 MiB per image and the request body to
48 MiB when vision is enabled. The native reader handles image decoding;
unlike Python with Pillow, this server does not convert other formats or apply
EXIF orientation. DeepSeek image input is not supported. Real vision model
inference has not yet been validated.

An interrupted generation or engine protocol error discards the engine
process. The next request starts it again; the failed request is not retried.
If a failed startup's child cannot be confirmed stopped, further restarts are
disabled to avoid loading a second engine.
A SIGINT or SIGTERM stops the server and its engine, including during engine
startup. An optional `--owner-fd` pipe also stops them if its writer closes;
on POSIX the server must be in its own process group. This guard alone does
not provide the managed launch ready-file or attestation protocol. POSIX
owner loss sends SIGTERM to the process group and escalates to SIGKILL after
45 seconds if the server is still running. A descendant that ignores SIGTERM
could survive if the server exits first. On Windows, tree containment needs
the launcher-owned job object.
A matched stop string sends STOP and drains the current request when the engine
supports it. An older engine without STOP runs to the end of the request.

A short real-model smoke test passed on an RTX 3080 Ti with the Qwen3.8
IQ3_S pack and a 4,096-token context: text and streaming on all three APIs,
a chat tool call, and a stop followed by another request. The Python server
matched the text, tool-call arguments and stop result on those prompts. This
does not verify the missing features above or full parity.
Forced chat and Anthropic tool calls with thinking disabled also passed a
short real-Qwen smoke test on the same GPU and context size.
The installed orca Q4_K_M tokenizer matched Python on four sample inputs,
and its template matched Python on three sample conversations, including a
tool declaration. With a fake engine, its tokenizer and template served
text requests on all three APIs. These checks did not load the model weights.
A separate real-model smoke test used the orca Q4_K_M config, its requested
262,144-token context, an RTX 3080 Ti, and Linux. One short non-streaming text
request succeeded on each API. The Go server process used 24,952 KiB RSS after
those requests; that excludes the native engine and model and is above 25 MB
(decimal). This single run did not establish full parity or a memory ceiling.

On Linux with a Ryzen 7 9800X3D, five model-free runs using the
Swift Heretic IQ4_XS tokenizer and a fake engine had a median Go server RSS of
25,480 KiB after one short Chat request. With `--no-responses --no-anthropic`,
the median was 25,372 KiB. The 19,073,089-byte binary and these RSS figures
exclude the engine and model. They do not establish a memory ceiling or a
reliable memory saving from disabling routes.

In separate model-free comparisons with that tokenizer, Go matched Python's
token IDs for 2,000 generated texts, each tested with special-token parsing
both on and off. Its template also matched Python's prompt bytes for 500
generated conversations covering tool declarations, tool replies and
reasoning settings. These samples do not establish full parity.

The later 19,075,123-byte binary was measured over five runs per setting,
after one streaming and one non-streaming Chat request in each run. Median Go
server RSS was 25,472 KiB with all routes and 25,536 KiB with the optional
routes disabled. The engine was fake and model weights were not loaded.
