"""The Sakshi-COS kernel.

The integration layer ("cognitive operating system") that the third project
calls for. It owns the persistent cognitive state, wires the observer +
controller into the agent loop, checkpoints after every run, and supports
resuming a session from disk. This is the single entry point a product or a
real LLM agent would build against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..agent.controller import ControllerConfig, ReliabilityController
from ..agent.loop import SakshiAgent, Worker
from ..agent.observer import MetaCognitiveObserver
from ..core.embeddings import Embedder, HashingEmbedder
from ..core.llm import LLM
from ..core.state import CognitiveState
from .persistence import CognitiveStore


@dataclass
class SakshiKernel:
    store_root: str = "runs/sessions"
    embedder: Embedder = field(default_factory=HashingEmbedder)
    llm: Optional[LLM] = None
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    max_steps: int = 40

    def __post_init__(self) -> None:
        self.store = CognitiveStore(self.store_root)
        self.observer = MetaCognitiveObserver(embedder=self.embedder, llm=self.llm)
        self.controller = ReliabilityController(self.controller_config)

    def boot(self, session_id: str, goal: str,
             subgoals: Optional[List[str]] = None, resume: bool = True) -> CognitiveState:
        """Load an existing session if present, else create a fresh one."""
        if resume and self.store.exists(session_id):
            return self.store.load(session_id)
        return CognitiveState.new(session_id, goal, subgoals)

    def run(self, state: CognitiveState, worker: Worker, checkpoint: bool = True) -> CognitiveState:
        agent = SakshiAgent(worker=worker, observer=self.observer,
                            controller=self.controller, max_steps=self.max_steps)
        agent.run(state)
        if checkpoint:
            self.store.save(state)
        return state

    def status(self, state: CognitiveState) -> dict:
        last = state.observer.latest()
        return {
            "session_id": state.session_id,
            "goal": state.goal.goal,
            "current_focus": state.goal.current_focus,
            "steps": state.step_count,
            "beliefs": len(state.beliefs.beliefs),
            "memory_items": len(state.memory.items),
            "untrusted_memory": sum(1 for it in state.memory.items if not it.trusted),
            "pending_verifications": len(state.verification.pending),
            "latest_assessment": last.to_dict() if last else None,
        }
