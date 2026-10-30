"""Line-oriented differential oracle. Run with the project's .venv Python."""
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from serve.frontend import call_end, function_end, param_end, parse_tool_call

def number_tree(value):
    if isinstance(value, str):
        return {"string": [ord(c) for c in value]}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return {"integer": str(value)}
    if isinstance(value, float):
        return {"float_bits": struct.pack(">d", value).hex()}
    if isinstance(value, list):
        return [number_tree(item) for item in value]
    return {"object": sorted(
        [[number_tree(key), number_tree(item)] for key, item in value.items()],
        key=lambda pair: pair[0]["string"],
    )}

for line in sys.stdin:
    request = json.loads(line)
    text = request["text"]
    try:
        if request["op"] in ("parse", "parse_numbers"):
            call = parse_tool_call(text, request.get("schema"))
            result = {"name": call.name, "arguments": call.arguments}
            result = number_tree(result)
        elif request["op"] == "param":
            result = param_end(text, request["final"])
        elif request["op"] == "function":
            result = function_end(text)
        else:
            result = call_end(text)
        print(json.dumps({"result": result}, ensure_ascii=True), flush=True)
    except (ValueError, TypeError, AttributeError) as exc:
        print(json.dumps({"error": type(exc).__name__}), flush=True)
