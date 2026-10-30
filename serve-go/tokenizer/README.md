# Native pre-tokenizer

This standalone Go 1.22 module implements the two pre-tokenizer splits from
`tools/strata_tokenizer.py`. `tokenizer.Qwen35(text)` and
`tokenizer.DeepSeek3(text)` return half-open UTF-8 byte spans and an error.
Spans are ordered, nonempty and cover the input exactly. Both reject invalid
UTF-8 with `ErrInvalidUTF8`; the reference accepts Unicode strings, not
arbitrary byte sequences. Empty text has no spans.

This is only the pre-tokenizer. It does not load a vocabulary or model,
perform BPE, or recognize special tokens. DeepSeek uses three sequential
passes, including the unmatched pieces left by each pass.

The Unicode classes are generated from Python `regex` 2026.9.10 (Unicode
17), rather than the older `unicode` tables shipped with Go 1.22. The table
contains the six masks for `L`, `M`, `N`, `P`, `S`, and `\s`. It is part of
the Go source; neither Python nor `regex` is needed at runtime.

To regenerate the table from the repository root, using the reference `.venv`,
write to a temporary file in the same directory and replace the table only
after the generator succeeds:

```sh
.venv/bin/python -c '
import os, pathlib, subprocess, sys, tempfile

target = pathlib.Path("serve-go/tokenizer/unicode_table_generated.go")
fd, temp = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
try:
    with os.fdopen(fd, "wb") as output:
        subprocess.run(
            [sys.executable, "serve-go/tokenizer/generate_unicode.py"],
            stdout=output, check=True,
        )
    os.chmod(temp, 0o644)
    os.replace(temp, target)
finally:
    if os.path.exists(temp):
        os.unlink(temp)
'
```

On Windows PowerShell, use the same command but replace `.venv/bin/python`
with `& .venv\Scripts\python.exe`. The temporary file is on the same filesystem
as the target. `os.replace` uses an atomic replacement where supported; if
generation or replacement fails, the original table is left in place.

The generator checks both the `regex` version and the SHA-256 of the
1,114,112 class-mask bytes, including surrogate code points, before emitting
anything. `go test` also checks that digest for every code point.

Run the tests with `go test ./...` in this directory. When the repository's
`.venv/bin/python` or `.venv/Scripts/python.exe` is present, the tests
additionally compare every Unicode mask and a deterministic synthetic corpus
against the actual Python reference. Set `STRATA_REFERENCE_PYTHON` to another
Python with `regex` 2026.9.10 to select it explicitly; otherwise these two
reference tests are skipped if neither interpreter is present. No vocabulary,
model, GPU, network, or engine is used.
