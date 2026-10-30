"""Test-only bridge to the real Python pre-tokenizer and Unicode properties."""

import json
from pathlib import Path
import sys

import regex

if regex.__version__ != "2026.9.10":
    sys.exit(f"expected regex 2026.9.10, found {regex.__version__}")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import strata_tokenizer as reference  # noqa: E402


def spans(pieces):
    out = []
    pos = 0
    for piece in pieces:
        end = pos + len(piece.encode("utf-8"))
        out.append([pos, end])
        pos = end
    return out


if sys.argv[1] == "split":
    qwen = regex.compile(reference.QWEN35_PATTERN)
    deepseek = [regex.compile(pattern) for pattern in reference.DEEPSEEK3_PATTERNS]
    results = []
    # JSON from Go is UTF-8 even when Windows uses a different text encoding.
    for text in json.load(sys.stdin.buffer):
        q = qwen.findall(text)
        d = reference._sequence_findall(deepseek, text)
        if "".join(q) != text or "".join(d) != text:
            raise ValueError("reference did not cover input")
        results.append({"qwen": spans(q), "deepseek": spans(d)})
    json.dump(results, sys.stdout, separators=(",", ":"))
elif sys.argv[1] == "unicode":
    text = "".join(map(chr, range(0x110000)))
    masks = bytearray(len(text))
    patterns = (r"\p{L}+", r"\p{M}+", r"\p{N}+", r"\p{P}+", r"\p{S}+", r"\s+")
    for bit, pattern in enumerate(patterns):
        for match in regex.finditer(pattern, text):
            for codepoint in range(match.start(), match.end()):
                masks[codepoint] |= 1 << bit
    sys.stdout.buffer.write(masks)
else:
    sys.exit("expected split or unicode")
