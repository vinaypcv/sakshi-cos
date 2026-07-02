"""SakshiBench task suite.

Each task pairs a goal + a scripted worker that fails in one mode + a checker
that scores the final cognitive state. Tasks are the unit of evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from ..agent.loop import Worker
from ..core.state import CognitiveState
from ..core.types import FailureMode
from . import workers


@dataclass
class Task:
    task_id: str
    goal: str
    failure_mode: FailureMode
    make_worker: Callable[[], Worker]
    check: Callable[[CognitiveState], float]   # success in [0, 1]
    max_steps: int = 12


def _check_goal_drift(state: CognitiveState) -> float:
    ans = state.history[-1].worker_output if state.history else ""
    return 1.0 if "completed summary" in ans else 0.0


def _check_memory(state: CognitiveState) -> float:
    ans = state.history[-1].worker_output if state.history else ""
    return 1.0 if "total = 75" in ans else 0.0   # 40 + 35, pollution excluded


def _check_overconfidence(state: CognitiveState) -> float:
    ans = state.history[-1].worker_output if state.history else ""
    return 1.0 if "verified assumptions" in ans else 0.0


def _check_conflict(state: CognitiveState) -> float:
    ans = state.history[-1].worker_output if state.history else ""
    return 1.0 if "consistent decision" in ans else 0.0


def default_suite() -> List[Task]:
    return [
        Task("drift-01", "write a summary of the q3 sales report",
             FailureMode.GOAL_DRIFT, workers.GoalDriftWorker, _check_goal_drift),
        Task("memory-01", "compute the project budget total",
             FailureMode.MEMORY_POLLUTION, workers.MemoryPollutionWorker, _check_memory),
        Task("overconf-01", "build a delivery plan from validated assumptions",
             FailureMode.OVERCONFIDENCE, workers.OverconfidenceWorker, _check_overconfidence),
        Task("conflict-01", "decide a retry policy for the api client",
             FailureMode.BELIEF_CONFLICT, workers.BeliefConflictWorker, _check_conflict),
    ]
