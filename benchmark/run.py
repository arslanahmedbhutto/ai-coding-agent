"""Run the benchmark: every task, for each model, graded by hidden tests.

    python -m benchmark.run --models llama-3.3-70b-versatile openai/gpt-oss-120b
"""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from benchmark.tasks import TASKS
from coding_agent.agent import build_agent, collect, make_llm
from coding_agent.workspace import Workspace

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "benchmark_results.json"


def grade(ws: Workspace, task) -> tuple[bool, str]:
    ws.write_file("_hidden_test.py", task.hidden_test)
    out = ws.run_tests("_hidden_test.py")
    return "EXIT CODE: 0" in out, out[-600:]


def run_model(model: str, repeats: int = 1) -> list[dict]:
    rows = []
    for rep in range(repeats):
        for task in TASKS:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Workspace(tmp, task.id)
                for name, content in task.starter.items():
                    ws.write_file(name, content)
                agent = build_agent(make_llm(model), ws)
                s = collect(agent, task.prompt, max_steps=20)
                passed, log = grade(ws, task)
                rows.append({"model": model, "task": task.id, "kind": task.kind, "repeat": rep, "passed": passed,
                             "tool_calls": s.tool_calls, "seconds": round(s.seconds, 1), "error": s.error,
                             "fake_tool_call": s.fake_tool_call, "malformed_calls": s.malformed_calls,
                             "loops_broken": s.loops_broken, "test_log": log})
                mark = "PASS" if passed else "FAIL"
                print(f"  {model:<32} {task.id:<12} {mark}  {s.tool_calls:>2} tool calls  {s.seconds:5.1f}s"
                      + (f"  [{s.error[:60]}]" if s.error else "") + ("  [text tool call]" if s.fake_tool_call else ""),
                      flush=True)
                time.sleep(2)   # be gentle with free-tier rate limits
    return rows


def main():
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in .env (see .env.example)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    existing = json.loads(OUT.read_text(encoding="utf-8"))["runs"] if OUT.exists() else []
    runs = [r for r in existing if r["model"] not in args.models]
    for m in args.models:
        print(f"== {m}", flush=True)
        runs += run_model(m, args.repeats)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "tasks": len(TASKS), "runs": runs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
