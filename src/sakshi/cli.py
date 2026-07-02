"""Unified command line: `python -m sakshi <command>`.

    demo        run the offline kernel demo (drift -> replan -> recover -> resume)
    bench       run the scripted SakshiBench suite + report
    scenarios   run the tool-using ReAct real-use-case scenarios + report
    ablate      run the control-action ablation with bootstrap CIs
    all         run bench + scenarios + ablate
"""
from __future__ import annotations

import argparse
import json
import os

from .bench.harness import compare, run_suite
from .bench.report import write_reports
from .bench.scenarios import run_scenarios
from .eval.ablation import ablation_table
from .obs.trace import RunRecorder


def _dump(obj) -> str:
    return json.dumps(obj, indent=2)


def cmd_bench(args) -> None:
    results = run_suite()
    paths = write_reports(results, args.out)
    rec = RunRecorder(args.out, "bench", config={"suite": "scripted", "arms": ["baseline", "sakshi"]})
    for arm, res in results.items():
        for tid, tr in res.traces.items():
            from .cos.persistence import state_from_dict
            rec.log_steps(state_from_dict(tr))
    rec.manifest(compare(results))
    print(_dump(compare(results)))
    print("artifacts:", {**paths, "manifest": rec.manifest_path})


def cmd_scenarios(args) -> None:
    results = run_scenarios()
    os.makedirs(args.out, exist_ok=True)
    summary = {arm: results[arm]["summary"] for arm in results}
    with open(os.path.join(args.out, "scenarios_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for arm in results:
        with open(os.path.join(args.out, f"scenarios_traces_{arm}.json"), "w") as f:
            json.dump(results[arm], f, indent=2)
    print(_dump(summary))
    print("artifacts in", args.out)


def cmd_ablate(args) -> None:
    table = ablation_table()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(_dump(table))


def cmd_demo(args) -> None:
    from .bench.workers import GoalDriftWorker
    from .cos.kernel import SakshiKernel
    kernel = SakshiKernel(store_root=os.path.join(args.out, "sessions"), max_steps=12)
    state = kernel.boot("demo-001", goal="write a summary of the q3 sales report", resume=False)
    kernel.run(state, GoalDriftWorker())
    for r in state.history:
        print(f"step {r.step:>2} [{r.action.value:8}] drift={r.assessment.goal_drift:.2f} "
              f"| {r.intervention_note}")
    print("status:", _dump(kernel.status(state)))


def cmd_scale(args) -> None:
    from .eval.scale import render_markdown, run_scale
    from .eval.cost import project_cost
    import json as _json

    if args.dry_run:
        print(_dump(project_cost(args.model, args.n, arms=2)))
        print("\nThis is a pre-flight estimate. Run a 5-task pilot to refine token assumptions.")
        return

    inner = None
    if args.live:
        from .core.llm import AnthropicLLM, CassetteLLM
        base = AnthropicLLM(model=args.model)
        inner = CassetteLLM(args.cassette, mode=args.cassette_mode, inner=base) if args.cassette else base

    r = run_scale(n_per_mode=args.n, inner_llm=inner, sim_latency=args.sim_latency,
                  model_name=args.model, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scale_report.json"), "w") as f:
        _json.dump(r, f, indent=2)
    md = render_markdown(r)
    with open(os.path.join(args.out, "scale_report.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"\nartifacts: {args.out}/scale_report.{{json,md}}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="sakshi", description="Sakshi-COS toolkit")
    ap.add_argument("--out", default="runs", help="output directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "bench", "scenarios", "ablate", "all"):
        sub.add_parser(name)
    sp = sub.add_parser("scale", help="scale evaluation with CIs, cost, calibration")
    sp.add_argument("--n", type=int, default=50, help="tasks per failure mode")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--model", default="claude-sonnet-4-6")
    sp.add_argument("--live", action="store_true", help="use a real Anthropic model")
    sp.add_argument("--cassette", default="", help="record/replay path")
    sp.add_argument("--cassette-mode", default="record", choices=["record", "replay"])
    sp.add_argument("--sim-latency", type=float, default=0.0, help="offline synthetic latency/call")
    sp.add_argument("--dry-run", action="store_true", help="print cost projection and exit")
    args = ap.parse_args()

    if args.cmd == "all":
        for fn in (cmd_bench, cmd_scenarios, cmd_ablate):
            fn(args)
    else:
        {"demo": cmd_demo, "bench": cmd_bench, "scenarios": cmd_scenarios,
         "ablate": cmd_ablate, "scale": cmd_scale}[args.cmd](args)


if __name__ == "__main__":
    main()
