"""Run the tool-using ReAct agent on the real-use-case scenarios.

Offline (default): a prompt-sensitive mock model drives the real agent, tools,
observer, and controller. Baseline fails; Sakshi recovers.

Live: set ANTHROPIC_API_KEY and pass --live to drive the SAME agent with a real
model. Wrap it in CassetteLLM to record/replay for reproducible evaluation.

    python examples/run_scenarios.py
    ANTHROPIC_API_KEY=sk-... python examples/run_scenarios.py --live
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.bench.scenarios import default_scenarios, run_scenarios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use a real Anthropic model")
    ap.add_argument("--cassette", default="", help="record/replay path for live runs")
    args = ap.parse_args()

    llm = None
    if args.live:
        from sakshi.core.llm import AnthropicLLM, CassetteLLM
        base = AnthropicLLM(model="claude-sonnet-4-6")
        llm = CassetteLLM(args.cassette, mode="record", inner=base) if args.cassette else base

    results = run_scenarios(default_scenarios(), llm=llm)
    for arm in ("baseline", "sakshi"):
        s = results[arm]["summary"]
        print(f"\n== {arm} ==  success_rate={s['success_rate']}  "
              f"interventions={s['total_interventions']}")
        for tid, e in results[arm]["per_task"].items():
            print(f"   {tid:16} {e['failure_mode']:16} success={e['success']} "
                  f"steps={e['steps']} interventions={e['intervention_count']}")


if __name__ == "__main__":
    main()
