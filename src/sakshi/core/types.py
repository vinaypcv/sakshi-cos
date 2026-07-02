"""Core value types shared across Sakshi Agent, SakshiBench, and Sakshi-COS.

These are intentionally dependency-free dataclasses/enums so every layer can
exchange them without importing heavy frameworks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ControlAction(str, Enum):
    """The four control decisions the reliability controller can emit."""

    CONTINUE = "continue"   # the agent is healthy; keep going
    VERIFY = "verify"       # check a claim / memory item before trusting it
    REFLECT = "reflect"     # pause and re-examine reasoning / assumptions
    REPLAN = "replan"       # the agent has drifted; rebuild the plan from the goal


class FailureMode(str, Enum):
    """The reliability failure modes Sakshi is designed to detect and mitigate."""

    GOAL_DRIFT = "goal_drift"
    MEMORY_POLLUTION = "memory_pollution"
    VERIFICATION_COLLAPSE = "verification_collapse"
    BELIEF_CONFLICT = "belief_conflict"
    OVERCONFIDENCE = "overconfidence"
    PLAN_INCOHERENCE = "plan_incoherence"


@dataclass
class Belief:
    """A single belief the agent currently holds."""

    claim: str
    confidence: float = 0.5          # [0, 1]
    support: List[str] = field(default_factory=list)
    verified: bool = False
    source_step: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "confidence": round(self.confidence, 4),
            "support": list(self.support),
            "verified": self.verified,
            "source_step": self.source_step,
        }


@dataclass
class MemoryItem:
    """A working-memory entry with provenance so pollution can be traced."""

    content: str
    provenance: str = "unknown"      # e.g. "observation", "tool", "injected"
    trusted: bool = True
    step: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provenance": self.provenance,
            "trusted": self.trusted,
            "step": self.step,
        }


@dataclass
class Assessment:
    """One observer assessment of the agent's cognitive state at a step."""

    step: int
    goal_drift: float                # [0, 1] higher = more drift
    uncertainty: float               # [0, 1] higher = less sure
    conflict_score: float            # [0, 1] higher = more contradiction
    unverified_load: float           # [0, 1] share of high-stakes unverified beliefs
    flagged: List[FailureMode] = field(default_factory=list)
    rationale: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "goal_drift": round(self.goal_drift, 4),
            "uncertainty": round(self.uncertainty, 4),
            "conflict_score": round(self.conflict_score, 4),
            "unverified_load": round(self.unverified_load, 4),
            "flagged": [f.value for f in self.flagged],
            "rationale": self.rationale,
        }


@dataclass
class StepRecord:
    """A full trace record for a single agent step: what happened + the decision."""

    step: int
    worker_output: str
    assessment: Assessment
    action: ControlAction
    intervention_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "worker_output": self.worker_output,
            "assessment": self.assessment.to_dict(),
            "action": self.action.value,
            "intervention_note": self.intervention_note,
        }
