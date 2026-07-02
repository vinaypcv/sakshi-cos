"""The reliability controller.

Given an observer `Assessment`, it decides the next control action. The policy
is explicit and tunable rather than a black box, which matters for an
evaluation/research system: every decision is explainable from thresholds.

Priority order (highest-severity intervention wins):
    REPLAN  > REFLECT > VERIFY > CONTINUE
Hysteresis prevents thrash: once an intervention fires, a cooldown blocks the
same intervention for a couple of steps so the agent gets a chance to recover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ..core.types import Assessment, ControlAction, FailureMode


@dataclass
class ControllerConfig:
    replan_drift: float = 0.5
    reflect_conflict: float = 0.3
    reflect_uncertainty: float = 0.65
    verify_unverified: float = 0.5
    cooldown_steps: int = 2
    enable_replan: bool = True
    enable_reflect: bool = True
    enable_verify: bool = True


@dataclass
class ReliabilityController:
    config: ControllerConfig = field(default_factory=ControllerConfig)
    _last_fired: Dict[ControlAction, int] = field(default_factory=dict)

    def decide(self, assessment: Assessment) -> Tuple[ControlAction, str]:
        step = assessment.step
        cfg = self.config

        def on_cooldown(action: ControlAction) -> bool:
            last = self._last_fired.get(action, -10_000)
            return (step - last) < cfg.cooldown_steps

        # REPLAN: the agent has materially drifted from the goal.
        if (cfg.enable_replan and assessment.goal_drift >= cfg.replan_drift
                and not on_cooldown(ControlAction.REPLAN)):
            return self._fire(ControlAction.REPLAN, step,
                              f"goal_drift {assessment.goal_drift:.2f} >= {cfg.replan_drift}")

        # REFLECT: contradictory beliefs or high uncertainty -> re-examine reasoning.
        if (cfg.enable_reflect
                and (assessment.conflict_score >= cfg.reflect_conflict
                     or assessment.uncertainty >= cfg.reflect_uncertainty)
                and not on_cooldown(ControlAction.REFLECT)):
            reason = (f"conflict {assessment.conflict_score:.2f}"
                      if assessment.conflict_score >= cfg.reflect_conflict
                      else f"uncertainty {assessment.uncertainty:.2f}")
            return self._fire(ControlAction.REFLECT, step, reason)

        # VERIFY: too much high-confidence unverified belief -> check before trusting.
        if (cfg.enable_verify and assessment.unverified_load >= cfg.verify_unverified
                and not on_cooldown(ControlAction.VERIFY)):
            return self._fire(ControlAction.VERIFY, step,
                              f"unverified_load {assessment.unverified_load:.2f} >= {cfg.verify_unverified}")

        # Memory pollution always at least triggers a verify if nothing stronger fired.
        if (cfg.enable_verify and FailureMode.MEMORY_POLLUTION in assessment.flagged
                and not on_cooldown(ControlAction.VERIFY)):
            return self._fire(ControlAction.VERIFY, step, "memory pollution detected")

        return ControlAction.CONTINUE, "all signals within bounds"

    def _fire(self, action: ControlAction, step: int, reason: str) -> Tuple[ControlAction, str]:
        self._last_fired[action] = step
        return action, reason
