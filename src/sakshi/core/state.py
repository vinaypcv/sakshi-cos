"""The unified cognitive state.

This is the data structure at the heart of Sakshi-COS. It is the persistent
"cognitive state" whose absence the project identifies as the root cause of
long-horizon agent failure. Every substate is serializable so a session can be
checkpointed and resumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import Assessment, Belief, MemoryItem, StepRecord


@dataclass
class GoalState:
    goal: str
    subgoals: List[str] = field(default_factory=list)
    # A short, evolving description of what the agent is *currently* focused on.
    current_focus: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "subgoals": list(self.subgoals),
                "current_focus": self.current_focus}


@dataclass
class BeliefState:
    beliefs: List[Belief] = field(default_factory=list)

    def add(self, belief: Belief) -> None:
        self.beliefs.append(belief)

    def unverified_high_conf(self, threshold: float = 0.7) -> List[Belief]:
        return [b for b in self.beliefs if not b.verified and b.confidence >= threshold]

    def to_dict(self) -> Dict[str, Any]:
        return {"beliefs": [b.to_dict() for b in self.beliefs]}


@dataclass
class MemoryState:
    items: List[MemoryItem] = field(default_factory=list)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def distrust(self, predicate) -> int:
        """Mark matching items untrusted; returns count affected."""
        n = 0
        for it in self.items:
            if predicate(it) and it.trusted:
                it.trusted = False
                n += 1
        return n

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [i.to_dict() for i in self.items]}


@dataclass
class UncertaintyState:
    overall: float = 0.0
    history: List[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        self.overall = value
        self.history.append(round(value, 4))

    def to_dict(self) -> Dict[str, Any]:
        return {"overall": round(self.overall, 4), "history": list(self.history)}


@dataclass
class VerificationState:
    pending: List[str] = field(default_factory=list)
    completed: List[Dict[str, Any]] = field(default_factory=list)

    def queue(self, claim: str) -> None:
        if claim not in self.pending:
            self.pending.append(claim)

    def resolve(self, claim: str, ok: bool, note: str = "") -> None:
        if claim in self.pending:
            self.pending.remove(claim)
        self.completed.append({"claim": claim, "ok": ok, "note": note})

    def to_dict(self) -> Dict[str, Any]:
        return {"pending": list(self.pending), "completed": list(self.completed)}


@dataclass
class ObserverState:
    assessments: List[Assessment] = field(default_factory=list)

    def record(self, a: Assessment) -> None:
        self.assessments.append(a)

    def latest(self) -> Optional[Assessment]:
        return self.assessments[-1] if self.assessments else None

    def to_dict(self) -> Dict[str, Any]:
        return {"assessments": [a.to_dict() for a in self.assessments]}


@dataclass
class CognitiveState:
    """Top-level persistent cognitive state for a single agent session."""

    session_id: str
    goal: GoalState
    beliefs: BeliefState = field(default_factory=BeliefState)
    memory: MemoryState = field(default_factory=MemoryState)
    uncertainty: UncertaintyState = field(default_factory=UncertaintyState)
    verification: VerificationState = field(default_factory=VerificationState)
    observer: ObserverState = field(default_factory=ObserverState)
    history: List[StepRecord] = field(default_factory=list)
    step_count: int = 0

    @classmethod
    def new(cls, session_id: str, goal: str, subgoals: Optional[List[str]] = None) -> "CognitiveState":
        return cls(
            session_id=session_id,
            goal=GoalState(goal=goal, subgoals=subgoals or [], current_focus=goal),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "step_count": self.step_count,
            "goal": self.goal.to_dict(),
            "beliefs": self.beliefs.to_dict(),
            "memory": self.memory.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "verification": self.verification.to_dict(),
            "observer": self.observer.to_dict(),
            "history": [r.to_dict() for r in self.history],
        }
