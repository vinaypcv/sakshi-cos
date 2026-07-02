"""Report generation for SakshiBench: JSON artifacts + a markdown comparison."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from .harness import BenchResult, compare


def write_reports(results: Dict[str, BenchResult], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    cmp = compare(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(cmp, f, indent=2)
    paths["summary"] = os.path.join(out_dir, "summary.json")

    # Full traces, one file per arm, consumed by the dashboard.
    for arm, res in results.items():
        p = os.path.join(out_dir, f"traces_{arm}.json")
        with open(p, "w") as f:
            json.dump({"arm": arm, "per_task": res.per_task, "traces": res.traces,
                       "summary": res.summary}, f, indent=2)
        paths[f"traces_{arm}"] = p

    md = _markdown(cmp, results)
    p = os.path.join(out_dir, "report.md")
    with open(p, "w") as f:
        f.write(md)
    paths["report"] = p
    return paths


def _markdown(cmp: Dict[str, Any], results: Dict[str, BenchResult]) -> str:
    b, s, d = cmp["baseline"], cmp["sakshi"], cmp["delta"]
    lines = [
        "# SakshiBench Report",
        "",
        "Baseline = same agents with the meta-cognitive controller disabled. "
        "Sakshi = identical agents with goal-drift / verification / reflection control active.",
        "",
        "## Headline",
        "",
        "| Metric | Baseline | Sakshi | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Success rate | {b.get('success_rate')} | {s.get('success_rate')} | {d['success_rate']:+} |",
        f"| Mean final drift | {b.get('mean_final_drift')} | {s.get('mean_final_drift')} | {d['mean_final_drift']:+} |",
        f"| Recovery rate | {b.get('recovery_rate')} | {s.get('recovery_rate')} | {d['recovery_rate']:+} |",
        f"| Total interventions | {b.get('total_interventions')} | {s.get('total_interventions')} | — |",
        "",
        "## Per-task (Sakshi arm)",
        "",
        "| Task | Failure mode | Success | Peak drift | Final drift | Interventions | Recovered |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for tid, e in results["sakshi"].per_task.items():
        lines.append(
            f"| {tid} | {e['failure_mode']} | {e['success']} | {e['peak_drift']} | "
            f"{e['final_drift']} | {e['intervention_count']} | {'yes' if e['recovered'] else 'no'} |"
        )
    lines += ["", "## Per-task (Baseline arm)", "",
              "| Task | Failure mode | Success | Peak drift | Final drift |",
              "| --- | --- | ---: | ---: | ---: |"]
    for tid, e in results["baseline"].per_task.items():
        lines.append(
            f"| {tid} | {e['failure_mode']} | {e['success']} | {e['peak_drift']} | {e['final_drift']} |"
        )
    lines.append("")
    return "\n".join(lines)
