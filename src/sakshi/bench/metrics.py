"""Run-level reliability metrics.

Computed from a finished `CognitiveState` trace, independent of task success, so
they describe *how* a run behaved, not just whether it passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.state import CognitiveState
from ..core.types import ControlAction


@dataclass
class RunMetrics:
    success: float
    steps: int
    peak_drift: float
    final_drift: float
    mean_uncertainty: float
    peak_conflict: float
    intervention_count: int
    interventions: Dict[str, int]
    recovered: bool

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        for k in ("peak_drift", "final_drift", "mean_uncertainty", "peak_conflict", "success"):
            d[k] = round(d[k], 4)
        return d


def compute_run_metrics(state: CognitiveState, success: float) -> RunMetrics:
    assessments = state.observer.assessments
    drifts = [a.goal_drift for a in assessments] or [0.0]
    uncs = [a.uncertainty for a in assessments] or [0.0]
    confs = [a.conflict_score for a in assessments] or [0.0]

    counts: Dict[str, int] = {a.value: 0 for a in ControlAction}
    for rec in state.history:
        counts[rec.action.value] += 1
    intervention_count = sum(v for k, v in counts.items() if k != ControlAction.CONTINUE.value)

    # "Recovered": drift peaked above the replan threshold region but ended low.
    recovered = (max(drifts) >= 0.6 and drifts[-1] < 0.45) if len(drifts) > 1 else False

    return RunMetrics(
        success=success,
        steps=state.step_count,
        peak_drift=max(drifts),
        final_drift=drifts[-1],
        mean_uncertainty=sum(uncs) / len(uncs),
        peak_conflict=max(confs),
        intervention_count=intervention_count,
        interventions={k: v for k, v in counts.items() if k != ControlAction.CONTINUE.value},
        recovered=recovered,
    )


def aggregate(metrics: List[RunMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "n_tasks": n,
        "success_rate": round(sum(m.success for m in metrics) / n, 4),
        "mean_peak_drift": round(sum(m.peak_drift for m in metrics) / n, 4),
        "mean_final_drift": round(sum(m.final_drift for m in metrics) / n, 4),
        "mean_uncertainty": round(sum(m.mean_uncertainty for m in metrics) / n, 4),
        "total_interventions": sum(m.intervention_count for m in metrics),
        "recovery_rate": round(sum(1 for m in metrics if m.recovered) / n, 4),
    }
