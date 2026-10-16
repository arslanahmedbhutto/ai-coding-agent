"""Summarise benchmark results into a table and a chart.

    python -m benchmark.report
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "benchmark_results.json"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GREEN, RED = "#1baf7a", "#e34948"


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    s = runs.groupby("model").agg(passed=("passed", "sum"), runs=("passed", "size"),
                                  avg_tool_calls=("tool_calls", "mean"), avg_seconds=("seconds", "mean"),
                                  text_tool_calls=("fake_tool_call", "sum"), errors=("error", lambda e: (e != "").sum()),
                                  malformed_retries=("malformed_calls", "sum"), loops_broken=("loops_broken", "sum"))
    s["pass_rate_%"] = 100 * s["passed"] / s["runs"]
    return s.sort_values(["pass_rate_%", "avg_seconds"], ascending=[False, True])


def main():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    runs = pd.DataFrame(data["runs"])
    s = summary_table(runs)
    grid = runs.pivot_table(index="task", columns="model", values="passed", aggfunc="mean")
    grid = grid.reindex(columns=s.index)

    md = ["# Benchmark results", "",
          f"{data['tasks']} tasks, each graded by hidden pytest tests the agent never sees.", "",
          "| Model | Passed | Avg tool calls | Avg time (s) | Malformed calls retried | Loops broken | Run errors |",
          "|---|---|---|---|---|---|---|"]
    for m, r in s.iterrows():
        md.append(f"| {m} | {int(r.passed)}/{int(r.runs)} ({r['pass_rate_%']:.0f}%) | {r.avg_tool_calls:.1f} | "
                  f"{r.avg_seconds:.1f} | {int(r.malformed_retries)} | {int(r.loops_broken)} | {int(r.errors)} |")
    md += ["", "## Per task", "", "| Task | " + " | ".join(grid.columns) + " |", "|---|" + "---|" * len(grid.columns)]
    for task, row in grid.iterrows():
        md.append(f"| {task} | " + " | ".join("✅" if v == 1 else "❌" if v == 0 else f"{v:.0%}" for v in row) + " |")
    (ROOT / "reports" / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 10.5,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    models = list(s.index)[::-1]
    labels = [f"{m}\n{int(s.loc[m, 'passed'])}/{int(s.loc[m, 'runs'])} passed" for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.9 * len(models) + 1.6), sharey=True)
    axes[0].barh(labels, [s.loc[m, "avg_seconds"] for m in models], color="#2a78d6", height=0.55)
    for i, m in enumerate(models):
        axes[0].text(s.loc[m, "avg_seconds"] * 1.02, i, f"{s.loc[m, 'avg_seconds']:.0f}s", va="center", fontsize=9.5)
    axes[0].set_title("Average time per task (free tier)", pad=28)
    axes[0].set_xlabel("seconds")
    left = [0.0] * len(models)
    for col, color, lab in [("loops_broken", "#eb6834", "Repetition loops broken"),
                            ("malformed_retries", "#4a3aa7", "Malformed tool calls recovered")]:
        vals = [float(s.loc[m, col]) for m in models]
        axes[1].barh(labels, vals, left=left, color=color, height=0.55, label=lab, edgecolor=SURFACE, linewidth=1.5)
        left = [l + v for l, v in zip(left, vals)]
    for i, total in enumerate(left):
        axes[1].text(total + 0.2, i, f"{total:.0f}", va="center", fontsize=9.5)
    axes[1].set_title("Times the agent's safeguards stepped in", pad=28)
    axes[1].set_xlabel(f"interventions across {int(s['runs'].iloc[0])} tasks")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "benchmark.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
