"""Evaluation rigor: ablations + bootstrap confidence intervals.

Ablation isolates the contribution of each control action by disabling it (its
threshold pushed out of range) and re-running the deterministic suites. The
bootstrap CI gives an honest uncertainty band on the success rate given the
small task count.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..agent.controller import ControllerConfig, ReliabilityController
from ..agent.loop import SakshiAgent
from ..agent.observer import MetaCognitiveObserver
from ..core.embeddings import HashingEmbedder
from ..core.state import CognitiveState
from ..bench import metrics as bm
from ..bench.tasks import default_suite

_OFF = 9.9  # threshold value that can never be reached -> action disabled

ABLATIONS: Dict[str, ControllerConfig] = {
    "full": ControllerConfig(),
    "no_replan": ControllerConfig(enable_replan=False),
    "no_reflect": ControllerConfig(enable_reflect=False),
    "no_verify": ControllerConfig(enable_verify=False),
    "none": ControllerConfig(enable_replan=False, enable_reflect=False, enable_verify=False),
}


def _success_vector(config: ControllerConfig) -> List[float]:
    embedder = HashingEmbedder()
    out: List[float] = []
    for task in default_suite():
        state = CognitiveState.new(task.task_id, task.goal)
        agent = SakshiAgent(worker=task.make_worker(),
                            observer=MetaCognitiveObserver(embedder=embedder),
                            controller=ReliabilityController(config),
                            max_steps=task.max_steps)
        agent.run(state)
        out.append(task.check(state))
    return out


def bootstrap_ci(successes: List[float], n: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) 95% percentile bootstrap CI for the success rate."""
    if not successes:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    k = len(successes)
    means = []
    for _ in range(n):
        sample = [successes[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    mean = sum(successes) / k
    return mean, means[int(0.025 * n)], means[int(0.975 * n)]


def ablation_table() -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for name, cfg in ABLATIONS.items():
        sv = _success_vector(cfg)
        mean, lo, hi = bootstrap_ci(sv)
        rows[name] = {"success_rate": round(mean, 4),
                      "ci95": [round(lo, 4), round(hi, 4)],
                      "per_task": sv}
    return rows
