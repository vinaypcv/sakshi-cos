import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.agent.controller import ControllerConfig, ReliabilityController
from sakshi.agent import metrics
from sakshi.core.embeddings import HashingEmbedder, semantic_distance
from sakshi.core.state import CognitiveState
from sakshi.core.types import Assessment, Belief, ControlAction, MemoryItem
from sakshi.bench.harness import compare, run_suite


def _assessment(**kw):
    base = dict(step=5, goal_drift=0.0, uncertainty=0.0, conflict_score=0.0, unverified_load=0.0)
    base.update(kw)
    return Assessment(**base)


def test_embedder_deterministic_and_self_distance_zero():
    e = HashingEmbedder()
    assert semantic_distance(e, "the cat sat", "the cat sat") < 1e-9
    assert semantic_distance(e, "quarterly sales report", "office coffee budget") > 0.5


def test_goal_drift_increases_when_focus_shifts():
    e = HashingEmbedder()
    s = CognitiveState.new("t", "write a summary of the q3 sales report")
    assert metrics.goal_drift(s, e) < 0.05
    s.goal.current_focus = "optimizing the office coffee budget"
    assert metrics.goal_drift(s, e) > 0.4


def test_belief_conflict_detects_negation_pair():
    s = CognitiveState.new("t", "g")
    s.beliefs.add(Belief("the api is rate limited", 0.8))
    s.beliefs.add(Belief("the api is not rate limited", 0.8))
    assert metrics.belief_conflict(s.beliefs) > 0.0


def test_controller_priority_replan_over_others():
    c = ReliabilityController(ControllerConfig())
    a = _assessment(goal_drift=0.8, conflict_score=0.9, unverified_load=0.9)
    action, _ = c.decide(a)
    assert action is ControlAction.REPLAN


def test_controller_continue_when_clean():
    c = ReliabilityController(ControllerConfig())
    action, _ = c.decide(_assessment())
    assert action is ControlAction.CONTINUE


def test_controller_cooldown_prevents_thrash():
    c = ReliabilityController(ControllerConfig(cooldown_steps=2))
    a1 = _assessment(step=1, goal_drift=0.8)
    assert c.decide(a1)[0] is ControlAction.REPLAN
    a2 = _assessment(step=2, goal_drift=0.8)  # within cooldown
    assert c.decide(a2)[0] is not ControlAction.REPLAN


def test_harness_sakshi_beats_baseline():
    results = run_suite()
    cmp = compare(results)
    assert cmp["baseline"]["success_rate"] < cmp["sakshi"]["success_rate"]
    assert cmp["sakshi"]["success_rate"] == 1.0
    assert cmp["delta"]["success_rate"] > 0
