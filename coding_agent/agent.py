"""The coding agent: a small, explicit tool-calling loop (ReAct style) over the workspace tools.

Written by hand instead of using a prebuilt graph so that it can:
  * stream every step to the UI,
  * break repetition loops (same tool + same arguments again and again),
  * turn tool errors and rate limits into messages instead of crashes.
"""

import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from coding_agent.workspace import Workspace

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a careful Python coding agent working inside a workspace folder.

Rules:
- Use the tools. Never print a tool call as text; call the tool.
- Always write complete, runnable files with write_file.
- After writing code, run it (run_python) or test it (run_tests) to verify it really works.
- If something fails, read the error, fix the file and run it again. Repeat until it works.
- Before editing an existing file, read it first. Don't repeat a tool call you have already made with the same arguments.
- Finish with a short summary: what you built or fixed, and how you verified it."""

# Some models write tool calls as text instead of calling the tool
FAKE_TOOL_CALL = re.compile(r"<function[/=]|<tool_call>|\{\s*\"(name|filename)\"\s*:", re.IGNORECASE)
REPEAT_LIMIT = 2
MAX_BAD_CALLS = 3


class MalformedToolCall(Exception):
    """The provider rejected the model's tool call (e.g. Groq 'tool_use_failed')."""


@dataclass
class Step:
    kind: str                 # "tool_call", "tool_result", "message"
    name: str = ""
    args: dict = field(default_factory=dict)
    content: str = ""
    t: float = 0.0


@dataclass
class RunSummary:
    final: str = ""
    tool_calls: int = 0
    seconds: float = 0.0
    error: str = ""
    fake_tool_call: bool = False
    loops_broken: int = 0
    malformed_calls: int = 0
    steps: list = field(default_factory=list)


def make_llm(model: str = DEFAULT_MODEL, api_key: str | None = None, temperature: float = 0.0):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, api_key=api_key, max_retries=3)


class CodingAgent:
    def __init__(self, llm, workspace: Workspace, system_prompt: str = SYSTEM_PROMPT):
        self.tools = {t.name: t for t in workspace.tools()}
        self.llm = llm.bind_tools(list(self.tools.values()))
        self.system_prompt = system_prompt

    def _invoke(self, messages, attempts: int = 4):
        for i in range(attempts):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                msg = str(e)
                if "tool_use_failed" in msg or "Failed to call a function" in msg:
                    raise MalformedToolCall(msg) from e
                if "rate" not in msg.lower() or i == attempts - 1:   # rate limits on free tiers: wait and retry
                    raise
                time.sleep(8 * (i + 1))

    def stream(self, task: str, history: list | None = None, max_steps: int = 25):
        """Yield Step objects as the agent works; return a RunSummary."""
        summary, t0 = RunSummary(), time.time()
        msgs = [SystemMessage(self.system_prompt)]
        for role, text in history or []:
            msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
        msgs.append(HumanMessage(task))
        last_sig, repeats, bad_calls = None, 0, 0

        def emit(step):
            step.t = time.time() - t0
            summary.steps.append(step)
            return step

        try:
            while True:
                try:
                    ai = self._invoke(msgs)
                except MalformedToolCall:
                    bad_calls += 1
                    summary.malformed_calls += 1
                    if bad_calls > MAX_BAD_CALLS:
                        raise
                    yield emit(Step("message", content="↻ The model sent a malformed tool call; asking it to retry."))
                    msgs.append(HumanMessage("Your last tool call could not be parsed (invalid arguments). "
                                             "Call the tool again with valid JSON arguments."))
                    continue
                msgs.append(ai)
                text = ai.content if isinstance(ai.content, str) else str(ai.content)
                if text.strip():
                    summary.final = text
                    yield emit(Step("message", content=text))
                if not ai.tool_calls:
                    break
                if summary.tool_calls >= max_steps:
                    summary.error = f"Stopped after {max_steps} tool calls"
                    yield emit(Step("message", content=f"⚠️ {summary.error}"))
                    break
                for call in ai.tool_calls:
                    summary.tool_calls += 1
                    name, args = call["name"], call.get("args", {}) or {}
                    yield emit(Step("tool_call", name, args))
                    sig = (name, json.dumps(args, sort_keys=True))
                    repeats = repeats + 1 if sig == last_sig else 0
                    last_sig = sig
                    tool = self.tools.get(name)
                    try:
                        result = str(tool.invoke(args)) if tool else f"Error: unknown tool '{name}'"
                    except Exception as e:
                        result = f"Error running {name}: {type(e).__name__}: {e}"
                    if repeats >= REPEAT_LIMIT:
                        summary.loops_broken += 1
                        result += (f"\n\nNOTE: you have now called {name} with the same arguments {repeats + 1} times "
                                   "in a row and got the same result. Do something different: write the fix, run a "
                                   "different file, or finish with a summary.")
                    msgs.append(ToolMessage(content=result, tool_call_id=call["id"], name=name))
                    yield emit(Step("tool_result", name, content=result))
        except Exception as e:
            summary.error = f"{type(e).__name__}: {str(e)[:300]}"
            yield emit(Step("message", content=f"⚠️ {summary.error}"))
        summary.seconds = time.time() - t0
        summary.fake_tool_call = bool(FAKE_TOOL_CALL.search(summary.final or ""))
        return summary


def build_agent(llm, workspace: Workspace) -> CodingAgent:
    return CodingAgent(llm, workspace)


def run_task(agent: CodingAgent, task: str, history: list | None = None, max_steps: int = 25):
    return agent.stream(task, history=history, max_steps=max_steps)


def collect(agent: CodingAgent, task: str, **kw) -> RunSummary:
    gen = run_task(agent, task, **kw)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
