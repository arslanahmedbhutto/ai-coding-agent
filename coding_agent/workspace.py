"""A per-session workspace folder and the tools the agent uses inside it.

Safety model: every path is resolved and must stay inside the workspace, code runs in
a separate process with a timeout, and output is truncated. This protects against
accidents, not against deliberately malicious code: run untrusted tasks in a container.
"""

import io
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

from langchain_core.tools import StructuredTool

MAX_OUTPUT = 4000
RUN_TIMEOUT = 30


class Workspace:
    def __init__(self, root: Path | str, name: str | None = None):
        self.dir = (Path(root) / (name or uuid.uuid4().hex[:10])).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def path(self, filename: str) -> Path:
        p = (self.dir / filename).resolve()
        if p != self.dir and self.dir not in p.parents:
            raise ValueError(f"'{filename}' is outside the workspace")
        return p

    def files(self) -> list[str]:
        out = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                out.append(p.relative_to(self.dir).as_posix())
        return out

    def zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in self.files():
                z.write(self.dir / f, f)
        return buf.getvalue()

    def clear(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: list[str]) -> str:
        env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper() and "TOKEN" not in k.upper()}
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            r = subprocess.run(cmd, cwd=self.dir, capture_output=True, text=True, timeout=RUN_TIMEOUT,
                               env=env, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {RUN_TIMEOUT} seconds"
        out = f"EXIT CODE: {r.returncode}\n\nSTDOUT:\n{r.stdout.strip() or '(no output)'}"
        if r.stderr.strip():
            out += f"\n\nSTDERR:\n{r.stderr.strip()}"
        return out if len(out) <= MAX_OUTPUT else out[:MAX_OUTPUT] + "\n... (output truncated)"

    # ---------- tool implementations ----------
    def write_file(self, filename: str, content: str) -> str:
        p = self.path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {filename} ({content.count(chr(10)) + 1} lines)"

    def read_file(self, filename: str) -> str:
        p = self.path(filename)
        if not p.is_file():
            return f"Error: {filename} does not exist"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= MAX_OUTPUT * 3 else text[: MAX_OUTPUT * 3] + "\n... (truncated)"

    def list_files(self) -> str:
        return "\n".join(self.files()) or "(workspace is empty)"

    def run_python(self, filename: str) -> str:
        p = self.path(filename)
        if not p.is_file():
            return f"Error: {filename} does not exist"
        return self._run([sys.executable, str(p.relative_to(self.dir))])

    def run_tests(self, target: str = "") -> str:
        args = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
        if target:
            args.append(str(self.path(target).relative_to(self.dir)))
        return self._run(args)

    # ---------- LangChain tools ----------
    def tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                self.write_file, name="write_file",
                description="Create or overwrite a file in the workspace with the COMPLETE file content "
                            "(never a fragment). Subfolders are allowed."),
            StructuredTool.from_function(
                self.read_file, name="read_file", description="Read the full contents of a workspace file."),
            StructuredTool.from_function(
                self.list_files, name="list_files", description="List all files in the workspace."),
            StructuredTool.from_function(
                self.run_python, name="run_python",
                description="Run a Python file in the workspace. Returns exit code, stdout and stderr "
                            "(including tracebacks). Use it to test code and read errors."),
            StructuredTool.from_function(
                self.run_tests, name="run_tests",
                description="Run pytest in the workspace (optionally on one test file) and return the results."),
        ]
