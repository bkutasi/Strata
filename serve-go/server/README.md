# Experimental Go server

This is an opt-in experiment, not a replacement for the Python server. It is
not connected to `strata launch` or the desktop app. It accepts text-only Qwen
and DeepSeek V4.1 requests at `POST /v1/chat/completions`, with streaming and a
limited set of request fields, lists the model at `GET /v1/models`, and reports
health at `GET /health` and `GET /api/health`. It also accepts text-only
Anthropic requests at `POST /v1/messages`, including streaming, and counts
prompts at `POST /v1/messages/count_tokens` without running generation.
It also accepts stateless text-only requests at `POST /v1/responses`, with
named streaming events, reasoning and completed function calls. Stored responses,
images, structured output, and custom tools are not supported.
Non-finite generated tool arguments fail the response. Tool
argument object keys may be reordered by the parser.
The APIs support `auto` and `none` tool choices; forced or named choices and
Anthropic reasoning budgets are not supported. Incomplete tool calls are not
streamed. All APIs reject images and use the renderer limits below. Other
endpoints and model families are not implemented.
Chat sampling supports temperature, top-p, top-k, min-p, repetition,
frequency and presence penalties, penalty window, and seed.

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

An interrupted generation or engine protocol error discards the engine
process. The next request starts it again; the failed request is not retried.
If a failed startup's child cannot be confirmed stopped, further restarts are
disabled to avoid loading a second engine.
A matched stop string discards subsequent output but waits for the engine to
finish generating, so a stop does not save inference work. No real-model or
GPU validation is claimed.
