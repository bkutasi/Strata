# Experimental Go server

This is an opt-in experiment, not a replacement for the Python server. It is
not connected to `strata launch` or the desktop app. It accepts text-only Qwen
requests at `POST /v1/chat/completions`, with streaming and a limited set of
request fields. Other endpoints and model families are not implemented.

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

An interrupted generation or engine protocol error discards the engine
process; this experiment does not restart it. A matched stop string discards
subsequent output but waits for the engine to finish generating, so a stop
does not save inference work. No real-model or GPU validation is claimed.
