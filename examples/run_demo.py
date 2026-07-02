"""End-to-end demo of Sakshi-COS on a single long-horizon task.

Shows: the agent drifting, the observer detecting it, the controller issuing a
REPLAN, the worker recovering, and the kernel checkpointing + resuming the
session from disk. Runs fully offline (HashingEmbedder + scripted worker).

    python examples/run_demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.bench.workers import GoalDriftWorker
from sakshi.cos.kernel import SakshiKernel


def banner(text: str) -> None:
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def main() -> None:
    kernel = SakshiKernel(store_root="runs/sessions", max_steps=12)

    banner("BOOT SESSION")
    state = kernel.boot("demo-001", goal="write a summary of the q3 sales report", resume=False)
    print(f"session={state.session_id}  goal={state.goal.goal!r}")

    banner("RUN (watch for drift -> REPLAN -> recovery)")
    kernel.run(state, GoalDriftWorker())
    for rec in state.history:
        a = rec.assessment
        tag = "" if rec.action.value == "continue" else "  <== INTERVENTION"
        print(f"step {rec.step:>2} | drift={a.goal_drift:.2f} unc={a.uncertainty:.2f} "
              f"| {rec.action.value.upper():8} | {rec.intervention_note}{tag}")
        print(f"          focus={state.history[rec.step-1].worker_output[:60]!r}")

    banner("FINAL STATUS")
    status = kernel.status(state)
    for k, v in status.items():
        if k != "latest_assessment":
            print(f"  {k}: {v}")
    print(f"  outcome: {state.history[-1].worker_output}")
    print(f"  checkpoint: runs/sessions/{state.session_id}.json")

    banner("RESUME FROM DISK (persistent cognitive state)")
    resumed = kernel.boot("demo-001", goal="(ignored on resume)", resume=True)
    print(f"  reloaded {len(resumed.history)} steps, "
          f"{len(resumed.observer.assessments)} assessments, "
          f"focus={resumed.goal.current_focus!r}")


if __name__ == "__main__":
    main()
