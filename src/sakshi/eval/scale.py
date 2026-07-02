"""Scale evaluation.

Runs the procedurally generated suite (N tasks per failure mode) across the
baseline and Sakshi arms, then reports:

  * success per mode and overall with Wilson 95% intervals,
  * the paired (same-task) baseline->Sakshi improvement with a CI,
  * token cost and latency per arm,
  * detector quality: AUC of each observer signal for predicting task failure,
  * intervention precision/recall against would-fail tasks.

Offline it uses the difficulty-driven mock (pipeline + stats validation). Pass an
inner LLM (AnthropicLLM) to produce real numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..agent.controller import ControllerConfig, ReliabilityController
from ..agent.loop import SakshiAgent
from ..agent.observer import MetaCognitiveObserver
from ..agent.react import LLMReActWorker
from ..core.embeddings import Embedder, HashingEmbedder
from ..core.llm import LLM, MeteredLLM, Usage
from ..core.state import CognitiveState
from ..core.types import Assessment, ControlAction
from ..bench.generator import MODES, generate
from ..bench.scenarios import FnLLM
from . import cost as costmod
from . import stats


@dataclass
class TaskOutcome:
    task_id: str
    mode: str
    difficulty: float
    success: float
    steps: int
    interventions: int
    peak_drift: float
    mean_uncertainty: float
    peak_conflict: float
    peak_unverified: float


def _signals(state: CognitiveState) -> Tuple[float, float, float, float]:
    a = state.observer.assessments
    if not a:
        return 0.0, 0.0, 0.0, 0.0
    return (max(x.goal_drift for x in a), sum(x.uncertainty for x in a) / len(a),
            max(x.conflict_score for x in a), max(x.unverified_load for x in a))


class _Passive:
    def decide(self, a: Assessment) -> Tuple[ControlAction, str]:
        return ControlAction.CONTINUE, "baseline"


def _run_arm(tasks, make_ctrl, inner_llm: Optional[LLM], sim_latency: float,
             embedder: Optional[Embedder] = None):
    outcomes: List[TaskOutcome] = []
    usage = Usage()
    observer_embedder = embedder or HashingEmbedder()
    for t in tasks:
        inner = inner_llm if inner_llm is not None else FnLLM(t.mock_model_fn)
        model = MeteredLLM(inner, sim_latency=sim_latency)
        state = CognitiveState.new(t.task_id, t.goal)
        SakshiAgent(worker=LLMReActWorker(llm=model, tools=t.build_tools()),
                    observer=MetaCognitiveObserver(embedder=observer_embedder),
                    controller=make_ctrl(), max_steps=t.max_steps).run(state)
        ans = state.history[-1].worker_output if state.history else ""
        pk_d, mn_u, pk_c, pk_v = _signals(state)
        interv = sum(1 for r in state.history if r.action is not ControlAction.CONTINUE)
        outcomes.append(TaskOutcome(t.task_id, t.failure_mode, t.difficulty, t.grade(ans),
                                    state.step_count, interv, pk_d, mn_u, pk_c, pk_v))
        usage.add(model.total)
    return outcomes, usage


def _by_mode(outcomes: List[TaskOutcome]) -> Dict[str, List[TaskOutcome]]:
    d: Dict[str, List[TaskOutcome]] = {m: [] for m in MODES}
    for o in outcomes:
        d[o.mode].append(o)
    return d


def run_scale(n_per_mode: int = 50, inner_llm: Optional[LLM] = None,
              sim_latency: float = 0.0, model_name: str = "claude-sonnet-5",
              seed: int = 7, embedder: Optional[Embedder] = None) -> Dict:
    tasks = generate(n_per_mode, seed)
    base, base_u = _run_arm(tasks, lambda: _Passive(), inner_llm, sim_latency, embedder)
    sak, sak_u = _run_arm(tasks, lambda: ReliabilityController(ControllerConfig()),
                          inner_llm, sim_latency, embedder)

    base_by, sak_by = _by_mode(base), _by_mode(sak)
    report: Dict = {"n_per_mode": n_per_mode, "n_tasks": len(tasks), "model": model_name,
                    "per_mode": {}, "overall": {}, "cost": {}, "calibration": {},
                    "intervention_quality": {}}

    # success + CIs per mode
    for m in MODES:
        bs = [o.success for o in base_by[m]]
        ss = [o.success for o in sak_by[m]]
        b_pt, b_lo, b_hi = stats.wilson_interval(int(sum(bs)), len(bs))
        s_pt, s_lo, s_hi = stats.wilson_interval(int(sum(ss)), len(ss))
        d_mean, d_lo, d_hi = stats.paired_diff_ci(ss, bs)
        report["per_mode"][m] = {
            "baseline": {"success": round(b_pt, 4), "ci95": [round(b_lo, 4), round(b_hi, 4)]},
            "sakshi": {"success": round(s_pt, 4), "ci95": [round(s_lo, 4), round(s_hi, 4)]},
            "paired_delta": {"mean": round(d_mean, 4), "ci95": [round(d_lo, 4), round(d_hi, 4)]},
        }

    # overall
    b_all = [o.success for o in base]
    s_all = [o.success for o in sak]
    bp, bl, bh = stats.wilson_interval(int(sum(b_all)), len(b_all))
    sp, sl, sh = stats.wilson_interval(int(sum(s_all)), len(s_all))
    dm, dl, dh = stats.paired_diff_ci(s_all, b_all)
    report["overall"] = {
        "baseline": {"success": round(bp, 4), "ci95": [round(bl, 4), round(bh, 4)]},
        "sakshi": {"success": round(sp, 4), "ci95": [round(sl, 4), round(sh, 4)]},
        "paired_delta": {"mean": round(dm, 4), "ci95": [round(dl, 4), round(dh, 4)]},
    }

    # cost / latency
    report["cost"] = {
        "baseline": costmod.summarize_cost(base_u, model_name, len(base)).to_dict(),
        "sakshi": costmod.summarize_cost(sak_u, model_name, len(sak)).to_dict(),
    }

    # detector calibration: does each signal predict failure? (baseline arm)
    labels = [1 if o.success == 0 else 0 for o in base]
    report["calibration"] = {
        "auc_goal_drift_vs_failure": round(stats.auc([o.peak_drift for o in base], labels), 4),
        "auc_uncertainty_vs_failure": round(stats.auc([o.mean_uncertainty for o in base], labels), 4),
        "auc_conflict_vs_failure": round(stats.auc([o.peak_conflict for o in base], labels), 4),
        "auc_unverified_vs_failure": round(stats.auc([o.peak_unverified for o in base], labels), 4),
        "note": "AUC = P(signal higher on a failed task than a passed one); 0.5 = no signal.",
    }

    # intervention precision/recall vs would-fail (baseline failure) tasks
    base_fail = {o.task_id for o in base if o.success == 0}
    sak_interv = {o.task_id for o in sak if o.interventions > 0}
    tp = len(base_fail & sak_interv)
    prec = tp / len(sak_interv) if sak_interv else 0.0
    rec = tp / len(base_fail) if base_fail else 0.0
    report["intervention_quality"] = {
        "intervened_tasks": len(sak_interv), "would_fail_tasks": len(base_fail),
        "precision": round(prec, 4), "recall": round(rec, 4),
    }
    report["_outcomes"] = {"baseline": [o.__dict__ for o in base],
                           "sakshi": [o.__dict__ for o in sak]}
    return report


def render_markdown(r: Dict) -> str:
    L = [f"# Sakshi-COS Scale Report",
         f"\n{r['n_tasks']} tasks ({r['n_per_mode']}/mode x {len(MODES)} modes), "
         f"model `{r['model']}`. Arms run the SAME tasks (paired).\n",
         "## Success by failure mode (95% Wilson CI)\n",
         "| Mode | Baseline | Sakshi | Paired Δ (95% CI) |",
         "| --- | ---: | ---: | ---: |"]
    for m in MODES:
        pm = r["per_mode"][m]
        b, s, d = pm["baseline"], pm["sakshi"], pm["paired_delta"]
        L.append(f"| {m} | {b['success']:.2f} [{b['ci95'][0]:.2f},{b['ci95'][1]:.2f}] | "
                 f"{s['success']:.2f} [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}] | "
                 f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] |")
    ov = r["overall"]
    L.append(f"| **overall** | **{ov['baseline']['success']:.2f}** | "
             f"**{ov['sakshi']['success']:.2f}** | "
             f"**{ov['paired_delta']['mean']:+.2f} "
             f"[{ov['paired_delta']['ci95'][0]:+.2f},{ov['paired_delta']['ci95'][1]:+.2f}]** |")

    cb, cs = r["cost"]["baseline"], r["cost"]["sakshi"]
    L += ["\n## Cost & latency\n",
          "| Arm | Calls | In tok | Out tok | USD | USD/task | s/task |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
          f"| baseline | {cb['calls']} | {cb['input_tokens']} | {cb['output_tokens']} | "
          f"{cb['usd']:.4f} | {cb['usd_per_task']:.5f} | {cb['s_per_task']:.3f} |",
          f"| sakshi | {cs['calls']} | {cs['input_tokens']} | {cs['output_tokens']} | "
          f"{cs['usd']:.4f} | {cs['usd_per_task']:.5f} | {cs['s_per_task']:.3f} |",
          "\n_Pricing is configurable; verify current rates. Offline tokens are estimated._"]

    cal = r["calibration"]
    L += ["\n## Detector quality (signal -> failure, AUC)\n",
          "| Signal | AUC |", "| --- | ---: |",
          f"| goal_drift | {cal['auc_goal_drift_vs_failure']} |",
          f"| uncertainty | {cal['auc_uncertainty_vs_failure']} |",
          f"| conflict | {cal['auc_conflict_vs_failure']} |",
          f"| unverified_load | {cal['auc_unverified_vs_failure']} |",
          f"\n{cal['note']}"]

    iq = r["intervention_quality"]
    L += [f"\n## Intervention targeting\n",
          f"Precision {iq['precision']:.2f}, recall {iq['recall']:.2f} "
          f"({iq['intervened_tasks']} intervened, {iq['would_fail_tasks']} would-fail).\n"]
    return "\n".join(L)
