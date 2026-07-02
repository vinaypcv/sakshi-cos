"""Persistent cognitive state store.

The project's thesis is that agents fail because cognitive state is ephemeral.
Sakshi-COS fixes that by checkpointing the entire `CognitiveState` to disk and
rehydrating it on resume, so a long task can be paused, inspected, and continued
without losing goal, beliefs, memory, or observer history.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from ..core.state import (
    BeliefState,
    CognitiveState,
    GoalState,
    MemoryState,
    ObserverState,
    UncertaintyState,
    VerificationState,
)
from ..core.types import Assessment, Belief, ControlAction, FailureMode, MemoryItem, StepRecord


def _assessment_from_dict(d: Dict[str, Any]) -> Assessment:
    return Assessment(
        step=d["step"], goal_drift=d["goal_drift"], uncertainty=d["uncertainty"],
        conflict_score=d["conflict_score"], unverified_load=d["unverified_load"],
        flagged=[FailureMode(f) for f in d.get("flagged", [])],
        rationale=d.get("rationale", ""),
    )


def state_from_dict(d: Dict[str, Any]) -> CognitiveState:
    state = CognitiveState(
        session_id=d["session_id"],
        goal=GoalState(**d["goal"]),
        beliefs=BeliefState([Belief(**b) for b in d["beliefs"]["beliefs"]]),
        memory=MemoryState([MemoryItem(**m) for m in d["memory"]["items"]]),
        uncertainty=UncertaintyState(**d["uncertainty"]),
        verification=VerificationState(**d["verification"]),
        observer=ObserverState([_assessment_from_dict(a) for a in d["observer"]["assessments"]]),
        step_count=d.get("step_count", 0),
    )
    for r in d.get("history", []):
        state.history.append(StepRecord(
            step=r["step"], worker_output=r["worker_output"],
            assessment=_assessment_from_dict(r["assessment"]),
            action=ControlAction(r["action"]),
            intervention_note=r.get("intervention_note", ""),
        ))
    return state


class CognitiveStore:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.root, f"{session_id}.json")

    def exists(self, session_id: str) -> bool:
        return os.path.exists(self._path(session_id))

    def save(self, state: CognitiveState) -> str:
        path = self._path(state.session_id)
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        return path

    def load(self, session_id: str) -> CognitiveState:
        with open(self._path(session_id)) as f:
            return state_from_dict(json.load(f))
