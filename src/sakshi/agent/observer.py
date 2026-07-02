"""The meta-cognitive observer.

A separate module that watches the agent's cognitive state *indirectly* (it does
not share the worker's chain of thought) and produces an `Assessment`. This is
the "the agent observes itself through a second module" property called for in
the project plan. The observer combines deterministic numeric signals with an
optional LLM judgment so it degrades gracefully without a live model.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.embeddings import Embedder, HashingEmbedder
from ..core.llm import LLM
from ..core.state import CognitiveState
from ..core.types import Assessment, FailureMode
from . import metrics

_OBSERVER_SYSTEM = (
    "You are a meta-cognitive observer monitoring another AI agent. You do not "
    "do the task. You judge whether the agent is still on-goal, over-confident, "
    "internally contradictory, or relying on unverified claims. Be terse."
)


class MetaCognitiveObserver:
    def __init__(self, embedder: Optional[Embedder] = None, llm: Optional[LLM] = None,
                 drift_flag: float = 0.45, conflict_flag: float = 0.3,
                 unc_flag: float = 0.6, uvl_flag: float = 0.5,
                 suspicious_provenance: Optional[set] = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.llm = llm
        self.drift_flag = drift_flag
        self.conflict_flag = conflict_flag
        self.unc_flag = unc_flag
        self.uvl_flag = uvl_flag
        self.suspicious_provenance = suspicious_provenance or {
            "injected", "unknown", "context", "unverified", "hearsay",
            "web:untrusted", "tool:unknown",
        }

    def assess(self, state: CognitiveState) -> Assessment:
        drift, unc, conflict, uvl = metrics.compute_signals(state, self.embedder)

        flagged: List[FailureMode] = []
        if drift >= self.drift_flag:
            flagged.append(FailureMode.GOAL_DRIFT)
        if conflict >= self.conflict_flag:
            flagged.append(FailureMode.BELIEF_CONFLICT)
        if unc >= self.unc_flag:
            flagged.append(FailureMode.OVERCONFIDENCE if uvl >= self.uvl_flag
                           else FailureMode.VERIFICATION_COLLAPSE)
        if uvl >= self.uvl_flag and FailureMode.OVERCONFIDENCE not in flagged:
            flagged.append(FailureMode.OVERCONFIDENCE)
        if any(it.trusted and it.provenance in self.suspicious_provenance
               for it in state.memory.items):
            flagged.append(FailureMode.MEMORY_POLLUTION)

        rationale = (
            f"drift={drift:.2f} unc={unc:.2f} conflict={conflict:.2f} unverified={uvl:.2f}"
        )
        rationale += self._llm_note(state) if self.llm else ""

        return Assessment(
            step=state.step_count,
            goal_drift=drift,
            uncertainty=unc,
            conflict_score=conflict,
            unverified_load=uvl,
            flagged=flagged,
            rationale=rationale,
        )

    def _llm_note(self, state: CognitiveState) -> str:  # pragma: no cover - needs live LLM
        recent = state.history[-1].worker_output if state.history else ""
        prompt = (
            f"GOAL: {state.goal.goal}\n"
            f"CURRENT FOCUS: {state.goal.current_focus}\n"
            f"LAST STEP OUTPUT: {recent}\n"
            "In one short sentence, flag the single biggest reliability risk right now."
        )
        try:
            note = self.llm.complete(prompt, system=_OBSERVER_SYSTEM, max_tokens=80)
            return f" | observer: {note.strip()}"
        except Exception as exc:
            return f" | observer-error: {exc}"
