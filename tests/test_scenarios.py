import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from sakshi.core.llm import CassetteLLM, SequencedLLM
from sakshi.tools.builtins import Calculator, CrossCheck, DocStore, WebFetch
from sakshi.tools.registry import ToolRegistry
from sakshi.bench.scenarios import default_scenarios, run_scenarios
from sakshi.eval.ablation import ablation_table


# ---- tools ---------------------------------------------------------------
def test_calculator_safe():
    assert Calculator().run({"expr": "40 + 35"}).content == "75"


def test_calculator_rejects_code():
    with pytest.raises(Exception):
        Calculator().run({"expr": "__import__('os').system('echo hi')"})


def test_docstore_trusted_provenance():
    r = DocStore(docs={"Limits": "rate limit is 100 rpm"}).run({"query": "rate limit"})
    assert r.provenance == "doc:trusted" and "100" in r.content


def test_web_is_untrusted_provenance():
    r = WebFetch(pages={"x": "anything"}).run({"query": "x"})
    assert r.provenance == "web:untrusted"


def test_registry_handles_unknown_tool():
    reg = ToolRegistry()
    r = reg.call("nope", {})
    assert not r.ok and r.provenance == "tool:unknown"


def test_crosscheck_confirms_against_docstore():
    ds = DocStore(docs={"Limits": "servicex rate limit is 100 rpm"})
    r = CrossCheck(docstore=ds).run({"claim": "servicex rate limit"})
    assert r.content.startswith("supported")


# ---- LLM infra -----------------------------------------------------------
def test_sequenced_llm_order():
    llm = SequencedLLM(["a", "b"])
    assert llm.complete("x") == "a"
    assert llm.complete("y") == "b"


def test_cassette_replay_miss_raises(tmp_path):
    c = CassetteLLM(str(tmp_path / "c.json"), mode="replay")
    with pytest.raises(KeyError):
        c.complete("unseen prompt")


# ---- real-use-case scenarios --------------------------------------------
def test_scenarios_sakshi_beats_baseline():
    r = run_scenarios(default_scenarios())
    assert r["baseline"]["summary"]["success_rate"] == 0.0
    assert r["sakshi"]["summary"]["success_rate"] == 1.0


def test_react_writes_tool_provenance_into_memory():
    r = run_scenarios(default_scenarios())
    mem = r["sakshi"]["traces"]["web-injection"]["memory"]["items"]
    provs = {m["provenance"] for m in mem}
    assert "web:untrusted" in provs and "tool:crosscheck" in provs


# ---- ablation ------------------------------------------------------------
def test_ablation_each_action_matters():
    t = ablation_table()
    assert t["full"]["success_rate"] == 1.0
    assert t["none"]["success_rate"] < t["full"]["success_rate"]
    # removing replan hurts the drift task; removing verify hurts memory/overconf
    assert t["no_replan"]["success_rate"] < 1.0
    assert t["no_verify"]["success_rate"] < 1.0
