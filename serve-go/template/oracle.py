"""Prepare existing model-free golden inputs using Strata's Python frontend."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from serve.frontend import openai_to_messages, use_family

for case in json.loads((root / "serve/chat_golden.json").read_text()):
    kwargs = {k: case[k] for k in ("enable_thinking",) if k in case}
    print(json.dumps({"name": case["name"], "family": "qwen",
                      "messages": case["messages"], "tools": case.get("tools"),
                      "kwargs": kwargs, "generate": case.get("add_generation_prompt", True),
                      "expected": case["rendered"]}, ensure_ascii=False))

use_family("deepseek41")
for case in json.loads((root / "serve/fixtures/deepseek41/golden.json").read_text()):
    messages, tools, kwargs = openai_to_messages(case["request"])
    print(json.dumps({"name": case["id"] + "/" + case["mode"], "family": "deepseek",
                      "messages": messages, "tools": tools, "kwargs": kwargs,
                      "generate": True, "expected": case["text"]}, ensure_ascii=False))
