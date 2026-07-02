"""SakshiBench harness.

Runs every task twice — once with a passive controller (baseline, no
meta-cognition) and once with the real reliability controller (sakshi) — and
collects per-run metrics plus full traces for the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..agent.controller import ControllerConfig, ReliabilityController
from ..agent.loop import SakshiAgent
from ..agent.observer import MetaCognitiveObserver
from ..core.embeddings import HashingEmbedder
from ..core.state import CognitiveState
from ..core.types import Assessment, ControlAction
from . import metrics as bm
from .tasks import Task, default_suite


class _PassiveController:
    """Always CONTINUE: the observer still measures, but never intervenes."""

    def decide(self, assessment: Assessment) -> Tuple[ControlAction, str]:
        return ControlAction.CONTINUE, "baseline (no meta-cognition)"


def _run_one(task: Task, controller, embedder: HashingEmbedder) -> Tuple[CognitiveState, bm.RunMetrics]:
    state = CognitiveState.new(session_id=task.task_id, goal=task.goal)
    observer = MetaCognitiveObserver(embedder=embedder)
    agent = SakshiAgent(worker=task.make_worker(), observer=observer,
                        controller=controller, max_steps=task.max_steps)
    agent.run(state)
    success = task.check(state)
    return state, bm.compute_run_metrics(state, success)


@dataclass
class BenchResult:
    arm: str  # "baseline" or "sakshi"
    per_task: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    traces: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


def run_suite(tasks: List[Task] = None) -> Dict[str, BenchResult]:
    tasks = tasks or default_suite()
    embedder = HashingEmbedder()
    results: Dict[str, BenchResult] = {}

    arms = {
        "baseline": lambda: _PassiveController(),
        "sakshi": lambda: ReliabilityController(ControllerConfig()),
    }
    for arm, make_ctrl in arms.items():
        res = BenchResult(arm=arm)
        run_metrics: List[bm.RunMetrics] = []
        for task in tasks:
            state, m = _run_one(task, make_ctrl(), embedder)
            entry = {"failure_mode": task.failure_mode.value, **m.to_dict()}
            res.per_task[task.task_id] = entry
            res.traces[task.task_id] = state.to_dict()
            run_metrics.append(m)
        res.summary = bm.aggregate(run_metrics)
        results[arm] = res
    return results


def compare(results: Dict[str, BenchResult]) -> Dict[str, Any]:
    b, s = results["baseline"].summary, results["sakshi"].summary
    return {
        "baseline": b,
        "sakshi": s,
        "delta": {
            "success_rate": round(s.get("success_rate", 0) - b.get("success_rate", 0), 4),
            "mean_final_drift": round(s.get("mean_final_drift", 0) - b.get("mean_final_drift", 0), 4),
            "recovery_rate": round(s.get("recovery_rate", 0) - b.get("recovery_rate", 0), 4),
        },
    }
