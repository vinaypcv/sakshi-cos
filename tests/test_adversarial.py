import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sakshi.bench.adversarial import load_adversarial_scenarios
from sakshi.bench.scenarios import run_scenarios


def test_loads_adversarial_prompt_injection_dataset():
    scenarios = load_adversarial_scenarios("data/adversarial/prompt_injection.jsonl")
    assert len(scenarios) >= 8
    assert all(s.failure_mode == "memory_pollution" for s in scenarios)


def test_adversarial_sakshi_beats_baseline():
    scenarios = load_adversarial_scenarios("data/adversarial/prompt_injection.jsonl")
    r = run_scenarios(scenarios)
    assert r["baseline"]["summary"]["success_rate"] == 0.0
    assert r["sakshi"]["summary"]["success_rate"] == 1.0
