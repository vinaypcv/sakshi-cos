"""Cost accounting.

Prices change; the table below is illustrative and MUST be set to current rates
before trusting dollar figures (see Anthropic pricing). Everything is per-million
tokens. The cost code is exact; only the constants need updating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..core.llm import Usage

# USD per million tokens. ILLUSTRATIVE — verify and override for real reporting.
PRICING: Dict[str, Dict[str, float]] = {
    "claude-opus-4-8":   {"in": 15.0, "out": 75.0},
    "claude-sonnet-4-6": {"in": 3.0,  "out": 15.0},
    "claude-haiku-4-5":  {"in": 0.80, "out": 4.0},
    "_default":          {"in": 3.0,  "out": 15.0},
}


def cost_usd(usage: Usage, model: str) -> float:
    p = PRICING.get(model, PRICING["_default"])
    return usage.input_tokens / 1e6 * p["in"] + usage.output_tokens / 1e6 * p["out"]


@dataclass
class CostReport:
    model: str
    usage: Usage
    usd: float
    usd_per_task: float
    s_per_task: float

    def to_dict(self) -> dict:
        return {"model": self.model, **self.usage.to_dict(),
                "usd": round(self.usd, 4), "usd_per_task": round(self.usd_per_task, 5),
                "s_per_task": round(self.s_per_task, 3)}


def summarize_cost(total: Usage, model: str, n_tasks: int) -> CostReport:
    usd = cost_usd(total, model)
    n = max(1, n_tasks)
    return CostReport(model=model, usage=total, usd=usd,
                      usd_per_task=usd / n, s_per_task=total.latency_s / n)


def project_cost(model: str, n_tasks: int, arms: int = 2,
                 avg_turns: float = 3.0, in_tokens_per_turn: int = 600,
                 out_tokens_per_turn: int = 150) -> dict:
    """Pre-flight estimate so you know the bill before spending. Override the
    per-turn token assumptions with numbers from a 5-task pilot for accuracy."""
    calls = n_tasks * arms * avg_turns
    u = Usage(input_tokens=int(calls * in_tokens_per_turn),
              output_tokens=int(calls * out_tokens_per_turn), calls=int(calls))
    return {"model": model, "tasks": n_tasks, "arms": arms, "est_calls": int(calls),
            "est_input_tokens": u.input_tokens, "est_output_tokens": u.output_tokens,
            "est_usd": round(cost_usd(u, model), 2)}
