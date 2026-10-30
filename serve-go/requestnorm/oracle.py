"""Small JSON-lines oracle. Imports only the actual, model-free Python frontend."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve import frontend

for line in sys.stdin:
    case = json.loads(line)
    try:
        frontend.use_family(case["family"])
        messages, tools, kwargs = frontend.openai_to_messages(case["request"])
        if case.get("require"):
            frontend.require_messages(messages)
        out = {"result": {"messages": messages, "tools": tools, "kwargs": kwargs}}
    except Exception as exc:
        out = {"error": str(exc)}
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))
