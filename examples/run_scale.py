"""Run the scale evaluation.

Offline (pipeline + stats validation, no key):
    python examples/run_scale.py --n 50

Pre-flight cost estimate before spending:
    python examples/run_scale.py --dry-run --n 50

Live with a real model, recording a cassette for reproducibility:
    ANTHROPIC_API_KEY=sk-... python examples/run_scale.py --live --n 50 \
        --cassette runs/scale.cassette.json

Replay the recorded cassette later with no key and no spend:
    python examples/run_scale.py --live --n 50 \
        --cassette runs/scale.cassette.json --cassette-mode replay
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.eval.cost import project_cost
from sakshi.eval.scale import render_markdown, run_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cassette", default="")
    ap.add_argument("--cassette-mode", default="record", choices=["record", "replay"])
    ap.add_argument("--sim-latency", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    if args.dry_run:
        print(json.dumps(project_cost(args.model, args.n), indent=2))
        return

    inner = None
    if args.live:
        from sakshi.core.llm import AnthropicLLM, CassetteLLM
        base = AnthropicLLM(model=args.model)
        inner = (CassetteLLM(args.cassette, mode=args.cassette_mode, inner=base)
                 if args.cassette else base)

    r = run_scale(n_per_mode=args.n, inner_llm=inner, sim_latency=args.sim_latency,
                  model_name=args.model, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scale_report.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(render_markdown(r))


if __name__ == "__main__":
    main()
