import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.bench.generator import MODES, generate
from sakshi.core.llm import Usage
from sakshi.eval import cost, stats
from sakshi.eval.scale import run_scale


# ---- generator -----------------------------------------------------------
def test_generate_balanced_and_gradable():
    tasks = generate(n_per_mode=8, seed=1)
    assert len(tasks) == 8 * len(MODES)
    counts = {m: 0 for m in MODES}
    for t in tasks:
        counts[t.failure_mode] += 1
        assert t.grade(f"answer: {t.ground_truth}") == 1.0
        if t.wrong_value:
            assert t.grade(f"answer: {t.wrong_value}") == 0.0
    assert all(c == 8 for c in counts.values())


def test_generate_is_deterministic():
    a = [(t.task_id, t.ground_truth) for t in generate(5, seed=3)]
    b = [(t.task_id, t.ground_truth) for t in generate(5, seed=3)]
    assert a == b


# ---- stats ---------------------------------------------------------------
def test_wilson_interval_bounds():
    p, lo, hi = stats.wilson_interval(50, 100)
    assert abs(p - 0.5) < 1e-9 and lo < 0.5 < hi and 0 <= lo and hi <= 1
    p0, lo0, hi0 = stats.wilson_interval(0, 20)
    assert p0 == 0.0 and lo0 == 0.0 and hi0 > 0.0


def test_paired_diff_ci_sign():
    mean, lo, hi = stats.paired_diff_ci([1, 1, 1, 0], [0, 0, 0, 0])
    assert mean == 0.75 and lo < mean < hi or lo <= mean <= hi


def test_auc_perfect_separation():
    assert stats.auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert abs(stats.auc([0.5, 0.5], [1, 0]) - 0.5) < 1e-9


# ---- cost ----------------------------------------------------------------
def test_cost_usd_matches_pricing():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    c = cost.cost_usd(u, "claude-sonnet-4-6")
    assert abs(c - (3.0 + 15.0)) < 1e-9


def test_project_cost_positive():
    p = cost.project_cost("claude-sonnet-4-6", 50)
    assert p["est_usd"] > 0 and p["est_calls"] == 300


# ---- scale (offline) -----------------------------------------------------
def test_scale_sakshi_beats_baseline_with_separated_ci():
    r = run_scale(n_per_mode=25, sim_latency=0.0)
    ov = r["overall"]
    assert ov["sakshi"]["success"] > ov["baseline"]["success"]
    # paired improvement CI should exclude zero at this n
    assert r["overall"]["paired_delta"]["ci95"][0] > 0
    assert set(r["per_mode"].keys()) == set(MODES)
    assert r["intervention_quality"]["recall"] > 0.5
