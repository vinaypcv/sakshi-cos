"""The Sakshi Agent loop.

Wraps any *base worker* (the thing that actually does the task) with the
meta-cognitive control layer. The worker is deliberately a thin protocol so the
same loop drives a scripted benchmark worker, a mock-LLM worker, or a real
tool-using LLM agent.

Loop per step:
    1. worker.step(state)            -> proposes output + updates to state
    2. observer.assess(state)        -> Assessment
    3. controller.decide(assessment) -> ControlAction
    4. if intervention: worker.on_directive(action, state) gets a chance to
       correct (replan back to goal, drop polluted memory, verify a claim...)
    5. record the step
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from ..core.state import CognitiveState
from ..core.types import Belief, ControlAction, MemoryItem, StepRecord
from .controller import ReliabilityController
from .observer import MetaCognitiveObserver


@dataclass
class WorkerStep:
    """What a worker returns each turn."""

    output: str
    focus: Optional[str] = None                 # new current_focus, if it shifted
    new_beliefs: List[Belief] = field(default_factory=list)
    new_memory: List[MemoryItem] = field(default_factory=list)
    done: bool = False


class Worker(Protocol):
    def step(self, state: CognitiveState) -> WorkerStep:
        ...

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        """React to a control intervention; return a short note. Optional logic."""
        ...


@dataclass
class SakshiAgent:
    worker: Worker
    observer: MetaCognitiveObserver = field(default_factory=MetaCognitiveObserver)
    controller: ReliabilityController = field(default_factory=ReliabilityController)
    max_steps: int = 30

    def run(self, state: CognitiveState) -> CognitiveState:
        for _ in range(self.max_steps):
            state.step_count += 1
            ws = self.worker.step(state)

            if ws.focus is not None:
                state.goal.current_focus = ws.focus
            for b in ws.new_beliefs:
                b.source_step = state.step_count
                state.beliefs.add(b)
            for m in ws.new_memory:
                m.step = state.step_count
                state.memory.add(m)

            assessment = self.observer.assess(state)
            state.observer.record(assessment)
            state.uncertainty.update(assessment.uncertainty)

            action, reason = self.controller.decide(assessment)
            note = reason
            if action is not ControlAction.CONTINUE:
                worker_note = self.worker.on_directive(action, state)
                note = f"{reason} -> {worker_note}" if worker_note else reason

            state.history.append(StepRecord(
                step=state.step_count,
                worker_output=ws.output,
                assessment=assessment,
                action=action,
                intervention_note=note,
            ))

            if ws.done and action is ControlAction.CONTINUE:
                break
        return state
