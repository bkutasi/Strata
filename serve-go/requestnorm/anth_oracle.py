"""JSON-lines oracle for the actual model-free Python frontend."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve import frontend

for line in sys.stdin:
    case = json.loads(line)
    request = case["request"]
    try:
        frontend.use_family(case["family"])
        if case.get("api") == "openai":
            messages, tools, kwargs = frontend.openai_to_messages(request)
        else:
            messages, tools, kwargs = frontend.anthropic_to_messages(
                request, case["think_unasked"]
            )
        if case.get("require"):
            frontend.require_messages(messages)
        out = {
            "result": {"messages": messages, "tools": tools, "kwargs": kwargs},
            "budget": request.get("reasoning_budget_tokens"),
        }
    except Exception as exc:
        out = {"error": str(exc)}
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))
