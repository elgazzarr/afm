"""Render an AFM-vs-published results table for the rebuttal.
Baseline numbers + protocols are fixed in
docs/superpowers/specs/2026-06-18-real-world-degradation-benchmarks-design.md §7.
Fill the AFM/FM rows from the script outputs, then render."""
from __future__ import annotations

COLS = ["method", "dataset", "metric", "value", "protocol"]

def render_results_table(rows: list[dict], fmt: str = "latex") -> str:
    if fmt == "markdown":
        head = "| " + " | ".join(COLS) + " |\n| " + " | ".join("---" for _ in COLS) + " |\n"
        body = "".join("| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |\n" for r in rows)
        return head + body
    # latex
    lines = ["\\begin{tabular}{lllll}", "\\toprule",
             " & ".join(COLS) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(r.get(c, "")) for c in COLS) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
