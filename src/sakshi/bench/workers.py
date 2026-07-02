"""Deterministic scripted workers for the benchmark.

Each worker is a *simulated* base agent engineered to fall into one specific
failure mode when left unsupervised. Crucially, it only self-corrects in
response to a controller directive (via `on_directive`). That lets the harness
make a fair, API-free comparison:

    baseline  = same worker, controller is passive (no directives delivered)
    sakshi    = same worker, real controller delivers VERIFY/REFLECT/REPLAN

Any difference in outcome is therefore attributable to the meta-cognitive layer,
not to two different agents. Real LLM workers implement the same `Worker`
protocol and plug straight into the harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.state import CognitiveState
from ..core.types import Belief, ControlAction, MemoryItem
from ..agent.loop import WorkerStep


class _Base:
    """Shared bookkeeping for scripted workers."""

    def __init__(self) -> None:
        self.i = 0
        self.corrected = False
        self.answer: Optional[str] = None

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:  # default
        return ""


class GoalDriftWorker(_Base):
    """Gets lured onto an off-goal tangent; REPLAN snaps it back to the goal."""

    DISTRACTOR = "optimizing the office coffee budget"

    def step(self, state: CognitiveState) -> WorkerStep:
        self.i += 1
        goal = state.goal.goal
        if self.i == 1:
            return WorkerStep(output=f"Planning approach to: {goal}", focus=goal)
        if self.i == 2:
            return WorkerStep(output="Gathering the relevant figures.",
                              focus=f"collecting data for {goal}")
        if self.i == 3 and not self.corrected:
            # Drift: a tangential subtopic hijacks the focus.
            return WorkerStep(
                output=f"Interesting side issue noticed: {self.DISTRACTOR}.",
                focus=self.DISTRACTOR)
        if self.i == 4 and not self.corrected:
            return WorkerStep(output=f"Deep-diving {self.DISTRACTOR} options.",
                              focus=f"detailed analysis of {self.DISTRACTOR}")
        # Either corrected, or baseline keeps drifting toward an off-goal finish.
        focus = goal if self.corrected else self.DISTRACTOR
        self.answer = (f"FINAL: completed summary of {goal}" if self.corrected
                       else f"FINAL: report on {self.DISTRACTOR}")
        return WorkerStep(output=self.answer, focus=focus, done=True)

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        if action is ControlAction.REPLAN:
            self.corrected = True
            state.goal.current_focus = state.goal.goal
            return "replanned: focus reset to original goal"
        return "acknowledged"


class MemoryPollutionWorker(_Base):
    """Ingests an injected false fact; VERIFY distrusts it before the final compute."""

    def step(self, state: CognitiveState) -> WorkerStep:
        self.i += 1
        if self.i == 1:
            return WorkerStep(output="Reading line items.",
                              new_memory=[MemoryItem("line A = 40", provenance="observation"),
                                          MemoryItem("line B = 35", provenance="observation")])
        if self.i == 2:
            # Pollution: a false, untrusted item is injected into memory.
            return WorkerStep(output="Picked up an extra note from context.",
                              new_memory=[MemoryItem("line A = 999", provenance="injected")])
        # Compute total. Trust only items still marked trusted.
        vals = {}
        for it in state.memory.items:
            if it.trusted and "=" in it.content:
                k, v = it.content.split("=")
                vals[k.strip()] = int(v.strip())  # last trusted write wins
        total = sum(vals.values())
        self.answer = f"FINAL: budget total = {total}"
        return WorkerStep(output=self.answer, done=True)

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        if action is ControlAction.VERIFY:
            n = state.memory.distrust(lambda it: it.provenance == "injected")
            return f"verified: distrusted {n} injected memory item(s)"
        return "acknowledged"


class OverconfidenceWorker(_Base):
    """Asserts a high-confidence but wrong unverified belief; VERIFY corrects it."""

    def step(self, state: CognitiveState) -> WorkerStep:
        self.i += 1
        if self.i == 1:
            return WorkerStep(
                output="Asserting the key assumption with high confidence.",
                new_beliefs=[Belief("the deadline is in 30 days", confidence=0.92)])
        if self.i == 2:
            return WorkerStep(
                output="Adding another confident assumption.",
                new_beliefs=[Belief("the client approved scope", confidence=0.9)])
        verified = all(b.verified for b in state.beliefs.beliefs)
        self.answer = ("FINAL: plan built on verified assumptions" if verified
                       else "FINAL: plan built on unverified assumptions")
        return WorkerStep(output=self.answer, done=True)

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        if action is ControlAction.VERIFY:
            for b in state.beliefs.beliefs:
                if not b.verified:
                    b.verified = True
                    b.support.append("checked against source-of-truth")
                    state.verification.resolve(b.claim, ok=True, note="verified by directive")
            return "verified: all outstanding assumptions checked"
        return "acknowledged"


class BeliefConflictWorker(_Base):
    """Holds two contradictory beliefs; REFLECT resolves the contradiction."""

    def step(self, state: CognitiveState) -> WorkerStep:
        self.i += 1
        if self.i == 1:
            return WorkerStep(output="Belief: the API is rate-limited.",
                              new_beliefs=[Belief("the api is rate limited", confidence=0.8)])
        if self.i == 2:
            return WorkerStep(output="Belief: the API is not rate-limited.",
                              new_beliefs=[Belief("the api is not rate limited", confidence=0.8)])
        consistent = not self._has_conflict(state)
        self.answer = ("FINAL: consistent decision on rate limiting" if consistent
                       else "FINAL: contradictory decision on rate limiting")
        return WorkerStep(output=self.answer, done=True)

    @staticmethod
    def _has_conflict(state: CognitiveState) -> bool:
        claims = [b.claim for b in state.beliefs.beliefs if b.confidence > 0]
        return ("the api is rate limited" in claims
                and "the api is not rate limited" in claims)

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        if action is ControlAction.REFLECT:
            # Resolve by retiring the lower-priority / negated belief.
            for b in state.beliefs.beliefs:
                if b.claim == "the api is not rate limited":
                    b.confidence = 0.0
                    b.support.append("retired during reflection")
            return "reflected: resolved contradictory beliefs"
        return "acknowledged"
