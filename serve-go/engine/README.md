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

This does not implement sampling options, token streaming callbacks, image
requests, log probabilities, batch slots, checkpoints or session commands,
telemetry, engine restart/recovery, GPU-aware silence detection, or Python
server feature parity. It does not start or configure an engine automatically.
Do not use the byte-only protocol fixture as a native engine.
