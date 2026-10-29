# Fixture HTTP tools v3

The recovered v2 tree was available. This complete source package fixes the confirmed parser mismatch and adds a fixture-only HTTP tool-call path. **Current Go compilation and execution remain unverified.**

## Changes

1. **Confirmed parser fix:** `parameterEnd` now returns the closer index for final empty/partial followers, including `x</parameter></fun` and `x</parameter><para`. Non-final partial followers still wait with -2. Complete follower boundaries retain their prior behavior. Direct parsing returns the completed value leniently; stream framing still rejects missing full function/tool-call closers.
2. **Validated requests:** supported OpenAI function definitions and Anthropic input schemas are normalized into declared parser schemas. Supported automatic, none, required/any and named choices are validated. Unknown shapes and unsupported schema keywords return 400 rather than silently degrading.
3. **Synthetic replay prompt:** tool requests carry an explicitly labeled `strata-byte-tools-v1` JSON transcript, schemas, normalized choice and forced prefix. Assistant calls and matching client-supplied results retain IDs, names and argument types. Orphan/duplicate/unresolved results and wrong-role blocks are rejected. Tools are not executed by this server.
4. **API-specific responses:** OpenAI nonstream tool_calls and indexed SSE tool_calls deltas; Anthropic nonstream tool_use blocks and separate block-start/input_json_delta/block-stop SSE events. Calls receive generated IDs once. Multiple calls, large integers and Unicode are preserved within the supported subset.
5. **Transactional publication:** tool-aware requests buffer/coalesce events until engine success, complete parsing and schema/choice validation. A later malformed call, engine error, EOF, pre-commit cancellation or length-truncated tool transaction cannot publish an earlier successful call. Initial SSE start frames/keepalives can precede an error. Partial network writes remain possible after commit; a terminal success marker is required.
6. **Offline exercise:** `exercise_tools.py` drives both mock and subprocess-fixture HTTP endpoints, streaming/nonstreaming, multiple calls, result replay, forced choices, none rejection and incomplete-output failures. It requires an already-built Go binary and performs only loopback traffic.

The engine bridge, FIFO, launcher, fake engine, baseline Python fixture tests, original endpoint exercise, module and old configs are byte-identical to v2. Python remains the launcher default. Native engines still fail the fixture capability check before GEN. Production tokenization/templates remain unavailable.

## Tests supplied

All **46 v2 Go test entrypoints** remain. There are now **64 entrypoints: 61 substantive tests and 3 subprocess helpers**. This adds 17 substantive tests and one helper. Every current Go test is **unrun** here.

New tests exercise the HTTP handler and actual subprocess bridge fixtures, not just parser helpers: request/schema validation, choice policies, distinct tool names, typed/large-number arguments, Unicode and marker strings, API-specific stream order, ID-preserving replay including reverse-ordered results, malformed output, EOF/engine errors, truncation and event-driven cancellation after a complete call but before DONE. Existing exhaustive parser split-point tests remain, and HTTP byte fixtures split every marker and UTF-8 sequence at byte boundaries. No fixed sleep or polling delay synchronizes these tests.

The old v2 assertion expecting -1 for a final `</fun` follower was corrected to 1. The previous assertion is archived under `v2-evidence/parser_v2_test_before_correction.txt`. No test was removed or skipped. Historical negative HTTP tests still reject their malformed definitions and required choices without tools; their old names do not describe the newly supported valid requests.

The shared Python reference corpus now has **39 cases**, including both confirmed mismatches. It is a partial reference reconstructed from supplied material, not the original complete frontend module.

## Actual execution in this sandbox

| Command / check | Result |
|---|---|
| `python3 excerpt_oracle.py` | Passed; 39 cases generated |
| `python3 verify.py` | **21 Python support/reference tests passed in 30.948 s**; overall exit **2** because Go is absent |
| `python3 -m py_compile *.py` | Passed, exit 0; new runner also compiled separately |
| `GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local go test -race -count=1 -timeout=180s ./...` | Command could not execute, exit **127**, `go: command not found`; Go tests NOT RUN |
| `python3 exercise.py --idle-seconds 1` | Refused absent binary, exit **2**; HTTP/memory NOT RUN |
| `python3 exercise_tools.py` | Refused absent binary, exit **2**; tool HTTP NOT RUN |
| Go build / vet / gofmt | NOT RUN, toolchain absent |
| Source delimiter, retained-test and unchanged-baseline checks | Passed; these are NOT Go parsing, compilation or semantic verification |
| Process check | No remaining Python fake-engine process found |
| Packaging metadata assembly | Initial NameError from an undefined baseline-path variable; corrected and rerun. See assembly-notes.txt |

No downloads or external network access were attempted. Python tests do not verify the Go implementation. Logs and `verification.json` retain the actual outcomes, including nonzero command exits. No current Go binary or memory measurement exists.

## User-supplied external v2 evidence, separate from this iteration

The user reports Go **1.22.2**, `go test -race -count=1 -timeout=90s ./...`, vet and trimpath build passed; **20** Python support/reference tests passed; `exercise.py --idle-seconds 1` passed isolated fixture HTTP checks; fixture RSS after workload and one second idle was **13,188 KiB**.

Those are **the user's external v2 results, not my runs**. They establish neither v3 correctness nor real-model serving, full Strata parity or production memory use. The user also found the additional final-partial-follower comparison failure, now corrected in source and covered by unrun Go regressions. See `v2-evidence/user_external_results.json` and `confirmed_mismatch.json`.

## Supported subset and unknowns

`ASSUMPTIONS.md` is the detailed contract. Important limits:

- Definitions require explicit object schemas. Supported keywords are type, description, properties, required, boolean additionalProperties, items and enum. This is not full JSON Schema. Integer validation is lexical, and enum numeric spellings are not normalized.
- Forced fixture scripts are continuations after the recorded XML prefix, not full calls. Forced-prefix whitespace and disabling the initial reasoning state remain fixture assumptions. There is no grammar-constrained inference.
- Stream arguments are complete payloads emitted after commit, not token-by-token argument deltas. API event structures are separately implemented fixture contracts, not exact upstream parity.
- History requires current declarations and string tool results. Rich result blocks, thinking/signature replay and persistent server sessions are unsupported.
- General quoted/fenced marker disambiguation, aliases, recovery, DSML and complete reasoning semantics remain incomplete. JSON fences inside explicit tool envelopes are supported; that does not establish general fence handling.
- Production models, tokenizer/Jinja templates, Responses, vision/MCP/images, sessions/replicas/split, manager ownership and upstream CLI/desktop/packaging integration remain missing. Browser UI remains excluded.
- No production 20 MiB result is claimed. The only new RSS figure in this report is the user's v2 fixture measurement. Transaction buffering adds memory whose real cost is unmeasured.

`COMPATIBILITY.md` and `compatibility.json` distinguish source implementation, Python fixture/reference execution, current Go execution, parity gaps and blockers.

## Offline build and independent verification

Extract into a new directory and enter `strata-go-experimental-v3`. Go 1.22+ and a C compiler for race testing must already be installed; Python 3.9+ is needed for support/exercise scripts. There are no third-party Go modules.

```sh
GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local go vet ./...
GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local go test -race -count=1 -timeout=180s ./...
GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local go build -trimpath -o strata-go-experimental .
python3 -m unittest -v test_fixture test_launch test_excerpt_oracle
python3 exercise.py --idle-seconds 1 --output endpoint-memory-v3.json
python3 exercise_tools.py --output tool-endpoints-v3.json

```

Each Go command disables module/toolchain downloads. The 180-second test timeout allows for the expanded subprocess/race suite; it is not synchronization. `python3 verify.py` is an alternative combined Python/gofmt/vet/race/build runner. The two HTTP exercises are separate commands and must also pass.

`exercise.py` retains its old fixture workload and optional Go-PID memory sampling. `exercise_tools.py` adds tool checks for both engine modes and both APIs, but does not measure memory or certify production parity. Neither runner has executed HTTP here.

For manual fixture launches, use `./strata-go-experimental --config tools-mock.json` or `./strata-go-experimental --config tools-protocol-fixture.json` from the extracted directory. The listener announcement supplies the ephemeral loopback address. Sample request/replay JSON files are included. `call_example` / `toolu_example` are illustrative fixture IDs, not observed generated calls.

## Next iteration

Run the v3 race/vet/build and both HTTP exercises first; fix any regression before expanding the surface. Then prioritize differential parser coverage for quoted/fenced markers, aliases and recovery, with explicit expected outputs. Keep real-model mode unavailable until tokenizer and template bytes can be established and compared. `FEEDBACK.txt` provides a concise result template.
