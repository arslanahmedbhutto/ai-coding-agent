"""AI Coding Agent: Streamlit interface.

    streamlit run app/streamlit_app.py
"""

import html
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from coding_agent.agent import DEFAULT_MODEL, build_agent, make_llm, run_task
from coding_agent.workspace import Workspace

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="AI Coding Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {margin-bottom: .2rem;}
.hero p {color: #6b6a65; margin-top: 0;}
.step {border-left: 3px solid #2a78d6; padding: .35rem .75rem; margin: .35rem 0; border-radius: 0 8px 8px 0;
       background: rgba(42,120,214,.06); font-size: .92rem;}
.step.run {border-color: #eb6834; background: rgba(235,104,52,.07);}
.step.ok {border-color: #1baf7a; background: rgba(27,175,122,.07);}
.step.err {border-color: #e34948; background: rgba(227,73,72,.07);}
.step details summary {cursor: pointer; opacity: .75; font-size: .85rem;}
pre.out {white-space: pre-wrap; font-size: .8rem; max-height: 320px; overflow: auto; margin: .3rem 0 0;
         padding: .5rem; border-radius: 6px; background: rgba(127,127,127,.10);}
.chip {display:inline-block; padding:.15rem .6rem; margin-right:.4rem; border-radius:999px; font-size:.8rem;
       background: rgba(127,127,127,.12);}
</style>
""", unsafe_allow_html=True)

ICONS = {"write_file": "✍️", "read_file": "📖", "list_files": "📂", "run_python": "▶️", "run_tests": "🧪"}
EXAMPLES = [
    ("🔢 Primes", "Create primes.py with is_prime(n) and primes_up_to(n), then print the primes up to 50 and run it."),
    ("🐞 Fix a bug", "__BUGGY__"),
    ("🧮 Calculator + tests", "Build calculator.py (add, subtract, multiply, divide; divide by zero raises ValueError) "
                             "and test_calculator.py with pytest tests. Run the tests."),
    ("📊 CSV report", "Create data.csv with 10 rows of fake sales (date, region, amount). Then write report.py that "
                     "prints total and average amount per region using only the standard library, and run it."),
    ("🎲 Simulation", "Write dice.py that simulates rolling two dice 10,000 times with a fixed seed and prints the "
                     "probability of each total from 2 to 12 as a text histogram. Run it."),
    ("🧹 Refactor", "Write messy.py with a long function that computes grade statistics, then refactor it into small "
                   "functions with docstrings, add pytest tests, and run them."),
]
BUGGY = """def average(numbers):
    total = 0
    for i in range(len(numbers) + 1):
        total += numbers[i]
    return total / len(numbers)


if __name__ == "__main__":
    print("Average:", average([10, 20, 30, 40]))
"""
FALLBACK_MODELS = [DEFAULT_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


# ---------- session state ----------
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


ss("session_id", uuid.uuid4().hex[:10])
ss("chat", [])            # list of {"role", "content", "steps", "stats"}
ss("history", [])         # (role, text) pairs passed back to the agent
ws = Workspace(Path(tempfile.gettempdir()) / "ai-coding-agent", st.session_state.session_id)


@st.cache_data(ttl=3600, show_spinner=False)
def groq_models(api_key: str) -> list[str]:
    try:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        r.raise_for_status()
        ids = sorted(m["id"] for m in r.json()["data"] if m.get("active", True))
        skip = ("whisper", "tts", "guard", "playai", "orpheus", "distil", "prompt", "allam")
        ids = [i for i in ids if not any(s in i.lower() for s in skip)]
        return sorted(ids, key=lambda i: (i != DEFAULT_MODEL, i)) or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        api_key = env_key
        st.success("Using the Groq key from this server's .env", icon="🔑")
    else:
        api_key = st.text_input("Groq API key", type="password", placeholder="gsk_...",
                                help="Stored only in this browser session. Free at console.groq.com/keys")
        st.caption("[Get a free key](https://console.groq.com/keys) · never stored on the server")
    models = groq_models(api_key) if api_key else FALLBACK_MODELS
    model = st.selectbox("Model", models, index=0)
    max_steps = st.slider("Max tool calls per task", 5, 40, 20)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.header("📁 Workspace")
    files = ws.files()
    st.caption(f"{len(files)} file(s)" if files else "Empty. The agent's files will appear here.")
    for f in files:
        st.markdown(f"`{f}`")
    uploads = st.file_uploader("Add files", accept_multiple_files=True, label_visibility="collapsed")
    if uploads:
        for up in uploads:
            ws.write_file(up.name, up.getvalue().decode("utf-8", errors="replace"))
        st.toast(f"Added {len(uploads)} file(s)")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Zip", ws.zip_bytes(), file_name="workspace.zip", disabled=not files,
                       use_container_width=True)
    if c2.button("🗑️ Reset", use_container_width=True):
        ws.clear()
        st.session_state.chat, st.session_state.history = [], []
        st.rerun()


# ---------- rendering helpers ----------
def lang_for(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {".py": "python", ".json": "json", ".md": "markdown", ".csv": "text", ".txt": "text",
            ".html": "html", ".js": "javascript", ".sql": "sql", ".yml": "yaml", ".yaml": "yaml"}.get(ext, "text")


def details(summary: str, body: str, open_: bool = False) -> str:
    """Collapsible block in plain HTML (Streamlit expanders can't be nested)."""
    return (f'<details{" open" if open_ else ""}><summary>{html.escape(summary)}</summary>'
            f'<pre class="out">{html.escape(body)}</pre></details>')


def render_step(step, container):
    if step.kind == "tool_call":
        icon = ICONS.get(step.name, "🔧")
        target = html.escape(str(step.args.get("filename") or step.args.get("target") or ""))
        css = "run" if step.name in ("run_python", "run_tests") else ""
        block = (f'<div class="step {css}">{icon} <b>{step.name}</b> <code>{target}</code> '
                 f'<span style="float:right;opacity:.6">{step.t:.1f}s</span>')
        if step.name == "write_file" and step.args.get("content"):
            block += details(f"View {step.args.get('filename', '')}", step.args["content"])
        container.markdown(block + "</div>", unsafe_allow_html=True)
    elif step.kind == "tool_result" and step.name in ("run_python", "run_tests"):
        ok = "EXIT CODE: 0" in step.content
        container.markdown(f'<div class="step {"ok" if ok else "err"}">{"✅ passed" if ok else "❌ failed"}'
                           + details("Output", step.content, open_=not ok) + "</div>", unsafe_allow_html=True)


def render_turn(turn, latest: bool = False):
    with st.chat_message(turn["role"], avatar="🧑‍💻" if turn["role"] == "user" else "🤖"):
        if turn["role"] == "user":
            st.markdown(turn["content"])
            return
        if turn.get("steps"):
            n = sum(s.kind == "tool_call" for s in turn["steps"])
            if latest:   # newest answer: show the full timeline
                st.caption(f"🔍 What the agent did ({n} tool calls)")
                for s in turn["steps"]:
                    render_step(s, st)
            else:
                with st.expander(f"🔍 {n} tool calls", expanded=False):
                    for s in turn["steps"]:
                        render_step(s, st)
        st.markdown(turn["content"] or "_(no summary)_")
        stats = turn.get("stats", {})
        if stats:
            st.markdown(f'<span class="chip">⏱ {stats["seconds"]:.1f}s</span>'
                        f'<span class="chip">🔧 {stats["tool_calls"]} tool calls</span>'
                        f'<span class="chip">🧠 {stats["model"]}</span>', unsafe_allow_html=True)


# ---------- header ----------
st.markdown('<div class="hero"><h1>🤖 AI Coding Agent</h1><p>Describe a task. The agent writes Python files, runs '
            'them, reads the errors and fixes its own code until it works.</p></div>', unsafe_allow_html=True)

tab_agent, tab_files, tab_bench = st.tabs(["💬 Agent", "📄 Files", "📊 Benchmark"])

# Chat box pinned to the bottom of the page (outside the tabs)
task = st.chat_input("Describe a coding task…", disabled=not api_key) or st.session_state.pop("pending", None)

with tab_agent:
    if not st.session_state.chat:
        st.write("**Try an example:**")
        cols = st.columns(3)
        for i, (label, prompt) in enumerate(EXAMPLES):
            if cols[i % 3].button(label, use_container_width=True, key=f"ex{i}"):
                if prompt == "__BUGGY__":
                    ws.write_file("average.py", BUGGY)
                    prompt = ("average.py crashes when run. Run it, read the traceback, fix the bug, make "
                              "average([]) return 0, and run it again to confirm.")
                st.session_state.pending = prompt
                st.rerun()

    for i, turn in enumerate(st.session_state.chat):
        render_turn(turn, latest=(i == len(st.session_state.chat) - 1 and not task))

    if not api_key:
        st.info("Add your Groq API key in the sidebar to start. It's free and only kept in this session.", icon="🔑")

    if task and api_key:
        st.session_state.chat.append({"role": "user", "content": task})
        render_turn(st.session_state.chat[-1])
        with st.chat_message("assistant", avatar="🤖"):
            steps = []
            with st.status("Agent is working…", expanded=True) as status:
                agent = build_agent(make_llm(model, api_key, temperature), ws)
                gen = run_task(agent, task, history=st.session_state.history, max_steps=max_steps)
                while True:
                    try:
                        step = next(gen)
                    except StopIteration as stop:
                        summary = stop.value
                        break
                    steps.append(step)
                    render_step(step, status)
                    if step.kind == "tool_call":
                        status.update(label=f"{ICONS.get(step.name, '🔧')} {step.name} "
                                            f"{step.args.get('filename', '')}…")
                state = "error" if summary.error else "complete"
                status.update(label=f"Done in {summary.seconds:.1f}s · {summary.tool_calls} tool calls",
                              state=state, expanded=False)
            final = summary.final
            if summary.fake_tool_call:
                final += ("\n\n> ⚠️ The model wrote a tool call as text instead of calling the tool, so that step "
                          "didn't actually run. Try again or pick another model.")
            st.markdown(final or "_(no summary)_")
        st.session_state.chat.append({"role": "assistant", "content": final, "steps": steps,
                                      "stats": {"seconds": summary.seconds, "tool_calls": summary.tool_calls,
                                                "model": model}})
        st.session_state.history += [("user", task), ("assistant", summary.final or "(no summary)")]
        st.rerun()

with tab_files:
    files = ws.files()
    if not files:
        st.info("No files yet. Ask the agent to build something, or upload files in the sidebar.")
    else:
        left, right = st.columns([1, 3])
        chosen = left.radio("Files", files, label_visibility="collapsed")
        with right:
            content = ws.read_file(chosen)
            edited = st.text_area(f"Edit `{chosen}`", content, height=320, key=f"edit-{chosen}")
            b1, b2, b3 = st.columns(3)
            if b1.button("💾 Save", use_container_width=True, disabled=edited == content):
                ws.write_file(chosen, edited)
                st.toast(f"Saved {chosen}")
                st.rerun()
            run_clicked = b2.button("▶️ Run", use_container_width=True, disabled=not chosen.endswith(".py"))
            test_clicked = b3.button("🧪 Run tests", use_container_width=True)
            if run_clicked:
                st.code(ws.run_python(chosen), language="text")
            if test_clicked:
                st.code(ws.run_tests(), language="text")
            with st.expander("Syntax-highlighted view", expanded=True):
                st.code(content, language=lang_for(chosen), line_numbers=True)

with tab_bench:
    path = ROOT / "reports" / "benchmark_results.json"
    if not path.exists():
        st.info("No benchmark results yet. Run `python -m benchmark.run --models <model>` locally.")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = pd.DataFrame(data["runs"])
        summary = (runs.groupby("model").agg(pass_rate=("passed", "mean"), tasks=("task", "count"),
                                             avg_tool_calls=("tool_calls", "mean"), avg_seconds=("seconds", "mean"))
                   .sort_values("pass_rate", ascending=False))
        summary["pass_rate"] = (100 * summary["pass_rate"]).round(0).astype(int).astype(str) + "%"
        st.subheader(f"{data['tasks']} tasks graded by hidden tests")
        st.dataframe(summary.round(1), use_container_width=True)
        st.dataframe(runs.pivot_table(index="task", columns="model", values="passed", aggfunc="mean")
                     .map(lambda v: "✅" if v == 1 else "❌" if v == 0 else f"{v:.0%}"), use_container_width=True)
        fig = ROOT / "reports" / "figures" / "benchmark.png"
        if fig.exists():
            st.image(str(fig))
