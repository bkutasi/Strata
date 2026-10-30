# Native engine subprocess client

This standalone Go module is an incremental component, not a Go serving stack.
Start a native executable with its `--serve` arguments, then call `Generate`
with token IDs and a maximum output count. `Start` accepts INFO and READY;
`Generate` reads T and DONE, ignores PP, RESUME and REUSED, and reports ERR. Stderr
goes to the supplied writer, never into the protocol. `Close` sends QUIT.
Calls are serialized. Context cancellation sends STOP when READY advertised it,
uses a bounded STOP write and waits up to one second for DONE, and discards the
process even on acknowledgement. A cancelled GEN write kills the child.
The process is also discarded on ERR, malformed or unsupported records, or EOF:
native errors may or may not have a trailing DONE, and fatal verification errors
must not be followed by another request. Callers must start a new engine.
DONE with a nonfinite finish returns ErrNonfinite and discards the process.
Startup and shutdown deadlines are supplied by the caller.
Failed startup waits up to three seconds for the killed child to be reaped;
`ErrUnreaped` means its exit could not be confirmed.

`Supervise` owns an initial engine and starts a replacement on the next request
after the child is discarded. It waits for the old child to be reaped and never
retries an in-flight generation. A restart with a different context limit is
rejected. If a failed startup returns `ErrUnreaped`, the supervisor closes
instead of risking a second model process.

`GenerateWithOptions` accepts typed sampling values and an optional synchronous
token callback. Nil sampling fields use native defaults (including greedy
temperature); zero temperature is greedy. Top-k 0 or above 64 maps to 64.
An active penalty uses a 64-token window unless a positive `PenaltyLastN` is
supplied. Seed 0 is omitted and the native engine chooses its seed. Invalid
values are rejected before writing. Callbacks run in order on the request
goroutine, after each token is collected and before DONE; they must return
promptly and must not reenter the same engine. A callback can return
`ErrStopGeneration` after a token to request STOP and drain DONE; the engine
can then be reused. If STOP is not supported, it drains the ongoing generation.
Other callback errors immediately kill and discard the child, returning the
error and tokens collected so far.
A callback panic also discards the child and propagates to the caller.
On cancellation no callbacks run after cancellation is observed; tokens drained
before DONE remain in the result. The caller must start a new engine after
either case.

This does not implement image requests, logit bias, log probabilities or
allowed-token constraints, batch slots, checkpoints or session commands,
telemetry, engine restart/recovery, GPU-aware silence detection, or Python
server feature parity (including tuning and prefill-only requests). It does not
start or configure an engine automatically.
Do not use the byte-only protocol fixture as a native engine.
