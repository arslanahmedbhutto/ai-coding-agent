"""Offline tests: the real LangGraph agent loop and real tools, driven by a scripted fake LLM."""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from benchmark.tasks import TASKS
from coding_agent.agent import FAKE_TOOL_CALL, build_agent, collect
from coding_agent.workspace import Workspace


class ScriptedLLM(BaseChatModel):
    """Returns pre-written AI messages in order (tool calls included)."""

    script: list[Any]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[self.i]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}])


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path, "w")


# ---------- workspace ----------
def test_paths_cannot_escape_workspace(ws):
    for bad in ["../evil.py", "..\\evil.py", "/etc/passwd", "sub/../../x.py"]:
        with pytest.raises(ValueError):
            ws.path(bad)
    assert ws.write_file("pkg/mod.py", "x = 1").startswith("Wrote")
    assert ws.files() == ["pkg/mod.py"]


def test_run_python_reports_tracebacks(ws):
    ws.write_file("ok.py", "print('hi')")
    ws.write_file("bad.py", "1/0")
    assert "EXIT CODE: 0" in ws.run_python("ok.py") and "hi" in ws.run_python("ok.py")
    out = ws.run_python("bad.py")
    assert "EXIT CODE: 1" in out and "ZeroDivisionError" in out
    assert "does not exist" in ws.run_python("missing.py")


def test_secrets_are_not_passed_to_executed_code(ws, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    ws.write_file("leak.py", "import os; print(os.getenv('GROQ_API_KEY'))")
    assert "gsk_secret" not in ws.run_python("leak.py")


def test_run_tests_and_zip(ws):
    ws.write_file("m.py", "def f():\n    return 2\n")
    ws.write_file("test_m.py", "from m import f\ndef test_f():\n    assert f() == 2\n")
    assert "EXIT CODE: 0" in ws.run_tests() and "1 passed" in ws.run_tests()
    assert len(ws.zip_bytes()) > 0


# ---------- agent loop ----------
def test_agent_writes_runs_and_fixes_code(ws):
    script = [
        call("write_file", {"filename": "avg.py", "content": "print(sum([1,2,3]) / 0)"}, 1),
        call("run_python", {"filename": "avg.py"}, 2),
        call("write_file", {"filename": "avg.py", "content": "print(sum([1,2,3]) / 3)"}, 3),
        call("run_python", {"filename": "avg.py"}, 4),
        AIMessage(content="Fixed the division by zero; avg.py now prints 2.0."),
    ]
    summary = collect(build_agent(ScriptedLLM(script=script), ws), "write avg.py")
    assert summary.error == "" and summary.tool_calls == 4
    results = [s.content for s in summary.steps if s.kind == "tool_result" and s.name == "run_python"]
    assert "ZeroDivisionError" in results[0] and "2.0" in results[1]
    assert summary.final.startswith("Fixed") and not summary.fake_tool_call
    assert ws.read_file("avg.py") == "print(sum([1,2,3]) / 3)"


def test_text_tool_calls_are_flagged(ws):
    script = [AIMessage(content='<function/run_python>{"filename": "buggy.py"}</function> I fixed it.')]
    summary = collect(build_agent(ScriptedLLM(script=script), ws), "fix buggy.py")
    assert summary.tool_calls == 0 and summary.fake_tool_call
    assert not FAKE_TOOL_CALL.search("All tests pass. I wrote primes.py and ran it.")


# ---------- benchmark tasks ----------
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_hidden_tests_fail_on_empty_or_starter_workspace(task, ws):
    for name, content in task.starter.items():
        ws.write_file(name, content)
    ws.write_file("_hidden_test.py", task.hidden_test)
    assert "EXIT CODE: 0" not in ws.run_tests("_hidden_test.py")


def test_repetition_loop_is_broken(ws):
    ws.write_file("a.py", "print(1)")
    script = [call("list_files", {}, i) for i in range(4)] + [AIMessage(content="Done.")]
    summary = collect(build_agent(ScriptedLLM(script=script), ws), "look around")
    notes = [s.content for s in summary.steps if s.kind == "tool_result"]
    assert summary.loops_broken >= 1 and "same arguments" in notes[-1]


def test_unknown_tool_and_step_limit(ws):
    script = [call("delete_everything", {}, 0)] + [call("list_files", {"x": i}, i + 1) for i in range(5)]
    summary = collect(build_agent(ScriptedLLM(script=script), ws), "go", max_steps=3)
    assert "unknown tool" in summary.steps[1].content
    assert summary.error.startswith("Stopped after 3")


def test_malformed_tool_call_is_retried(ws):
    class FlakyLLM(ScriptedLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.i == 0:
                self.i += 1
                raise RuntimeError("Error code: 400 - {'error': {'code': 'tool_use_failed', 'message': 'Failed to call a function'}}")
            return super()._generate(messages, stop, run_manager, **kwargs)

    script = [None, call("list_files", {}, 1), AIMessage(content="Listed the files.")]
    summary = collect(build_agent(FlakyLLM(script=script), ws), "list files")
    assert summary.error == "" and summary.malformed_calls == 1 and summary.final == "Listed the files."
