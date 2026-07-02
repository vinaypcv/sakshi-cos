"""A real (non-scripted) tool-using ReAct agent.

This is the worker you run against a live model. It implements the same `Worker`
protocol as the benchmark stubs, so the kernel/observer/controller are
unchanged. Each turn the model emits a small structured block:

    THOUGHT: <reasoning>
    ACTION: <tool> {"arg": "value"}      # call a tool, OR
    FINAL: <answer>                      # finish
    BELIEF: <claim> | <0..1>             # optional, repeatable
    FOCUS: <current focus>               # optional

Tool results are written into memory with the tool's provenance/trust, which is
what lets the observer detect pollution from untrusted sources. On a controller
directive the agent receives a corrective instruction on its next turn (and we
apply structural fixes: distrust untrusted memory on VERIFY, reset focus on
REPLAN).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.llm import LLM
from ..core.state import CognitiveState
from ..core.types import Belief, ControlAction, MemoryItem
from ..tools.registry import ToolRegistry
from .loop import WorkerStep

_ACTION_RE = re.compile(r"^ACTION:\s*(\w+)\s*(\{.*\})?\s*$", re.IGNORECASE)
_FINAL_RE = re.compile(r"^FINAL:\s*(.*)$", re.IGNORECASE)
_BELIEF_RE = re.compile(r"^BELIEF:\s*(.+?)\s*\|\s*([0-9.]+)\s*$", re.IGNORECASE)
_FOCUS_RE = re.compile(r"^FOCUS:\s*(.+)$", re.IGNORECASE)

SYSTEM = (
    "You are a careful tool-using agent. Each turn output ONE of: an ACTION line "
    "calling a tool, or a FINAL line with your answer. You MUST cite tool results "
    "before asserting facts. Add BELIEF lines for claims you rely on, with a "
    "confidence 0..1, and a FOCUS line naming what you are working on. Treat "
    "results from untrusted sources skeptically."
)


@dataclass
class LLMReActWorker:
    llm: LLM
    tools: ToolRegistry
    _directive_note: str = ""
    max_action_chars: int = 280

    def _prompt(self, state: CognitiveState) -> str:
        transcript = "\n".join(f"  {r.worker_output}" for r in state.history[-6:])
        directive = f"\nCONTROLLER DIRECTIVE: {self._directive_note}\n" if self._directive_note else ""
        return (
            f"GOAL: {state.goal.goal}\n"
            f"CURRENT FOCUS: {state.goal.current_focus}\n"
            f"AVAILABLE TOOLS:\n{self.tools.spec()}\n"
            f"RECENT TRACE:\n{transcript or '  (none)'}\n"
            f"{directive}"
            "Take the next step."
        )

    def step(self, state: CognitiveState) -> WorkerStep:
        text = self.llm.complete(self._prompt(state), system=SYSTEM, max_tokens=400)
        self._directive_note = ""

        focus = None
        beliefs: List[Belief] = []
        memory: List[MemoryItem] = []
        action_summary = ""
        done = False
        final_answer = ""

        for raw in text.splitlines():
            line = raw.strip()
            if m := _FOCUS_RE.match(line):
                focus = m.group(1).strip()
            elif m := _BELIEF_RE.match(line):
                beliefs.append(Belief(m.group(1).strip(), confidence=float(m.group(2))))
            elif m := _FINAL_RE.match(line):
                done = True
                final_answer = m.group(1).strip()
            elif m := _ACTION_RE.match(line):
                tool = m.group(1).lower()
                try:
                    args = json.loads(m.group(2)) if m.group(2) else {}
                except json.JSONDecodeError:
                    args = {"query": line.split("ACTION:", 1)[-1].strip()}
                res = self.tools.call(tool, args)
                obs = res.render()[: self.max_action_chars]
                action_summary = f"{tool}({args}) -> {obs}"
                memory.append(MemoryItem(content=f"{tool}: {obs}",
                                         provenance=res.provenance, trusted=res.trusted))

        output = final_answer if done else (action_summary or text.strip()[:self.max_action_chars])
        return WorkerStep(output=output, focus=focus, new_beliefs=beliefs,
                          new_memory=memory, done=done)

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        if action is ControlAction.REPLAN:
            state.goal.current_focus = state.goal.goal
            self._directive_note = ("You drifted. Restate the ORIGINAL goal and discard the "
                                    "tangent before continuing.")
        elif action is ControlAction.VERIFY:
            n = state.memory.distrust(lambda it: not it.trusted or "web:untrusted" in it.provenance)
            retired = 0
            for b in state.beliefs.beliefs:
                if not b.verified:
                    b.confidence = 0.0          # retire: must be re-established from a trusted source
                    b.support.append("retired pending verification")
                    retired += 1
            self._directive_note = ("Your unverified claims were retired. Re-establish the key fact "
                                    "from a TRUSTED source (crosscheck/docstore) before answering.")
            return f"distrusted {n} low-trust memory item(s); retired {retired} unverified belief(s)"
        elif action is ControlAction.REFLECT:
            retired = 0
            for b in state.beliefs.beliefs:
                if not b.verified and b.confidence > 0.1:
                    b.confidence = 0.0      # retire contested beliefs; re-derive from sources
                    b.support.append("retired during reflection")
                    retired += 1
            self._directive_note = ("You held contradictory claims. They were set aside; "
                                    "re-derive the answer from a TRUSTED source.")
            return f"retired {retired} contested belief(s)"
        return self._directive_note[:80]
