"""How to run Sakshi-COS with a REAL LLM agent.

This is the bridge from the offline/scripted demo to a live model. The worker
implements the exact same `Worker` protocol the benchmark uses, so the kernel,
observer, and controller are unchanged. It is not run by the test suite (it
needs ANTHROPIC_API_KEY); it is documentation-as-code.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-...
    python examples/run_with_llm.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.agent.loop import WorkerStep
from sakshi.core.llm import AnthropicLLM
from sakshi.core.state import CognitiveState
from sakshi.core.types import Belief, ControlAction, MemoryItem
from sakshi.cos.kernel import SakshiKernel

WORKER_SYSTEM = (
    "You are a task-executing agent. Each turn, take ONE concrete step toward the "
    "goal. Reply with: a 'STEP:' line describing what you did, an optional "
    "'FOCUS:' line naming your current focus, optional 'BELIEF: <claim> | <0-1 "
    "confidence>' lines, and 'DONE' on its own line when the goal is fully met."
)


class LLMWorker:
    """A minimal real-model worker. The observer/controller wrap it unchanged."""

    def __init__(self, llm: AnthropicLLM) -> None:
        self.llm = llm

    def step(self, state: CognitiveState) -> WorkerStep:
        transcript = "\n".join(f"- {r.worker_output}" for r in state.history[-6:])
        prompt = (
            f"GOAL: {state.goal.goal}\n"
            f"CURRENT FOCUS: {state.goal.current_focus}\n"
            f"RECENT STEPS:\n{transcript or '(none)'}\n\nTake the next step."
        )
        text = self.llm.complete(prompt, system=WORKER_SYSTEM, max_tokens=400)
        focus = None
        beliefs = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("FOCUS:"):
                focus = line.split(":", 1)[1].strip()
            elif line.upper().startswith("BELIEF:"):
                body = line.split(":", 1)[1]
                claim, _, conf = body.partition("|")
                try:
                    c = float(re.findall(r"[0-9.]+", conf)[0])
                except (IndexError, ValueError):
                    c = 0.6
                beliefs.append(Belief(claim.strip(), confidence=c))
        done = "DONE" in text.upper()
        return WorkerStep(output=text.strip()[:300], focus=focus,
                          new_beliefs=beliefs, done=done)

    def on_directive(self, action: ControlAction, state: CognitiveState) -> str:
        # Feed the intervention back to the model as a corrective instruction.
        directive = {
            ControlAction.REPLAN: "You have drifted. Re-state the ORIGINAL goal and "
                                  "discard the tangent before continuing.",
            ControlAction.REFLECT: "Pause. Identify and resolve any contradictory "
                                   "beliefs you currently hold.",
            ControlAction.VERIFY: "Verify your highest-confidence unverified claim "
                                  "against a source before relying on it.",
        }[action]
        note = self.llm.complete(
            f"GOAL: {state.goal.goal}\nINTERVENTION: {directive}\nRespond in one line.",
            system=WORKER_SYSTEM, max_tokens=120)
        if action is ControlAction.REPLAN:
            state.goal.current_focus = state.goal.goal
        if action is ControlAction.VERIFY:
            for b in state.beliefs.beliefs:
                b.verified = True
        return note.strip()[:120]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live demo."); return
    kernel = SakshiKernel(llm=AnthropicLLM(model="claude-sonnet-4-6"), max_steps=15)
    state = kernel.boot("llm-001", goal="produce a 5-line competitive analysis of two note-taking apps", resume=False)
    kernel.run(state, LLMWorker(kernel.llm))
    for r in state.history:
        print(f"step {r.step:>2} [{r.action.value:8}] {r.worker_output[:80]}")
    print("\nstatus:", kernel.status(state))


if __name__ == "__main__":
    main()
