# Experimental Go server

This is an opt-in experiment, not a replacement for the Python server. It is
not connected to `strata launch` or the desktop app. It accepts text-only Qwen
and DeepSeek V4.1 requests at `POST /v1/chat/completions`, with streaming and a
limited set of request fields, lists the model at `GET /v1/models` and
`GET /models`, and reports
health at `GET /health` and `GET /api/health`. It also accepts text-only
Anthropic requests at `POST /v1/messages`, including streaming, and counts
prompts at `POST /v1/messages/count_tokens` without running generation.
As in Python, requests naming another model use the loaded model and report
its name; model aliases are not supported.
It also accepts stateless text-only requests at `POST /v1/responses`, with
named streaming events, reasoning, stop strings and completed function calls. Stored responses,
images and custom tools are not supported.
As in the Python server, `store: true` is accepted but responses are not stored.
Codex client metadata is accepted; the optional Codex thread-title and
compaction-cache behaviors are not implemented.
Non-finite generated tool arguments fail the response. Tool
argument object keys may be reordered by the parser.
The APIs support `auto` and `none` tool choices. Chat and Anthropic requests
also support forced or named choices when thinking is disabled; forced choices
with thinking and positive reasoning budgets while thinking are not supported.
A zero or negative reasoning budget disables the limit; positive budgets are
accepted when thinking is disabled. As in the
Python Responses route, Responses `tool_choice: none` removes tools; other
choices are accepted but do not force a tool. Incomplete tool calls are not
streamed. All APIs reject images and use the renderer limits below. Other
endpoints and model families are not implemented.
Chat sampling supports temperature, top-p, top-k, min-p, repetition,
frequency and presence penalties, penalty window, and seed.
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

Build with Go 1.26.10:

```sh
go build -o strata-go-server ./cmd/strata-go-server
./strata-go-server -help
```

Run it only with a trusted model pack and template. The template renderer does
not provide Python's immutable sandbox, and rejects requests containing
literal control tokens rather than handling them like the Python server.
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
It supports `exe`, `cwd`, `args`, `tokenizer`, `model_name`, `host`, `api_key`,
integer `gpu`, string-valued `env`, and `log`. Other config fields are rejected
rather than silently ignored. `--api-key` overrides the config key, while
`--api-key-env` takes precedence over both. The config's host is used unless
`--listen` is supplied; check it before starting a network-accessible server.
This path has not been validated with every Python config.

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
