"""Bounded line-oriented oracle for frontend.OutputParser, not a server."""
import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve.frontend import OutputParser  # noqa: E402
from serve.test_native_tool_parser import CALL, TOOLS  # noqa: E402


def number_tree(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"string": [ord(c) for c in value]}
    if isinstance(value, int):
        return {"integer": str(value)}
    if isinstance(value, float):
        return {"float_bits": struct.pack(">d", value).hex()}
    if isinstance(value, list):
        return [number_tree(item) for item in value]
    return {"object": [[number_tree(key), number_tree(value[key])] for key in sorted(value)]}


def execute(request):
    if request.get("op") == "fixtures":
        specimens = json.loads((ROOT / "serve/fixtures/rcall_specimens.json").read_text())["specimens"]
        return {"call": CALL, "tools": TOOLS, "specimens": specimens}

    parser = OutputParser(thinking=request["thinking"], tools=request["tools"],
                          stream_tools=request["stream"], declared_only=request["declared_only"])
    ids = {}

    def norm(events):
        out = []
        for event in events:
            call = None
            if event.call is not None:
                ident = ids.setdefault(event.call.id, len(ids) + 1)
                call = {"name": event.call.name, "arguments": number_tree(event.call.arguments), "id": ident}
            out.append({"kind": event.kind, "text": event.text, "call": call})
        return out

    def status():
        return {"state": parser.state, "clean": parser.clean, "pending": len(parser.pending), "unread": parser.unread,
                "unread_names": parser.unread_names[:], "rescued": parser.rescued, "refused": parser.refused}

    batches, states = [], []
    for part in request["parts"]:
        batches.append(norm(parser.feed(part)))
        states.append(status())
    batches.append(norm(parser.finish(request["finish"] or None)))
    states.append(status())
    return {"batches": batches, "states": states}


for line in sys.stdin:
    try:
        print(json.dumps({"result": execute(json.loads(line))}, ensure_ascii=True), flush=True)
    except Exception as exc:  # The error itself is an oracle result, not a silent failure.
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=True), flush=True)
