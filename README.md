# Sakshi-COS

**Meta-cognitive infrastructure for reliable long-horizon AI agents.**

Long-horizon agents fail in characteristic ways: they drift off the goal, pollute their own
memory with untrusted content, stop verifying, hold contradictory beliefs, and get
overconfident. Sakshi-COS treats those as *measurable, controllable* phenomena. A separate
observer watches the agent's cognitive state, quantifies the risk, and a controller decides
whether to **continue, verify, reflect, or replan** — over a persistent cognitive state that
survives across sessions.

One codebase, the three projects from the plan, plus the engineering that makes it a portfolio:

| Layer | Package | What it is |
| --- | --- | --- |
| **1. Sakshi Agent** | `sakshi.agent` | Observer + metrics + controller + agent loop + real ReAct worker |
| **2. SakshiBench** | `sakshi.bench` | Failure-mode benchmark + tool-using scenarios + harness + report |
| **3. Sakshi-COS** | `sakshi.cos` | Kernel integrating everything over persistent, resumable state |
| Tools | `sakshi.tools` | Tool registry + safe tools, each carrying provenance/trust |
| Core / eval / obs | `sakshi.core`, `sakshi.eval`, `sakshi.obs` | State, LLM boundary, ablations, traces |

## Quick start (no API key needed)

```bash
export PYTHONPATH=src           # or: pip install -e ".[dev]"
python -m sakshi demo           # drift → REPLAN → recover → resume from disk
python -m sakshi bench          # scripted 4-mode benchmark + report
python -m sakshi scenarios      # tool-using ReAct agent on real-use-case scenarios
python -m sakshi ablate         # contribution of each control action (+ bootstrap CIs)
python -m sakshi all            # everything, into runs/
python -m pytest                # 18 tests
make all                        # same via Makefile
```

Open `apps/dashboard/index.html` and load `runs/traces_sakshi.json` or
`runs/scenarios_traces_sakshi.json` to see per-step cognitive telemetry.

## Results (deterministic)

| Suite | Baseline | Sakshi |
| --- | ---: | ---: |
| Scripted benchmark (drift / pollution / overconfidence / conflict) | 0.25 | **1.00** |
| ReAct scenarios (research-drift, web-injection) | 0.00 | **1.00** |

Ablation: removing **replan** drops the drift task, **reflect** the conflict task, **verify**
the memory task; `none` = 0.25, `full` = 1.00 — each control action recovers exactly what it
targets.

## Going live with a real model

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python examples/run_scenarios.py --live --cassette runs/scn.cassette.json
python examples/run_with_llm.py
```

The same `LLMReActWorker` + tools + observer + controller run live. `CassetteLLM` records the
model's responses once and replays them for reproducible evaluation and CI.

## How it works

```
worker.step(state)            propose next action + state updates (beliefs/memory w/ provenance)
observer.assess(state)        → {goal_drift, uncertainty, conflict, unverified_load}
controller.decide(assessment) → CONTINUE | VERIFY | REFLECT | REPLAN   (priority + cooldown)
worker.on_directive(action)   correct: refocus to goal, distrust untrusted memory, retire claims
kernel.store.save(state)      checkpoint; resume next session
```

The observer judges from cognitive state, not the worker's private reasoning — meta-cognition,
not self-grading. See `ARCHITECTURE.md` for the full design and `PORTFOLIO.md` for the
completion checklist, interview framing, and honest caveats.

## What is real vs. simulated

- **Real:** cognitive-state model, metrics, controller policy, persistence/resume, tool layer
  with provenance/trust, the ReAct agent, benchmark harness, ablation, observability, CLI/CI.
- **Swappable:** embedder (offline hashing → learned) and model (Mock/Sequenced/Cassette/Anthropic).
- **Simulated for deterministic evaluation:** the benchmark's base agents are scripted to fall
  into one failure mode each and self-correct only on a controller directive; the offline
  scenario model is a prompt-sensitive mock. Baseline and Sakshi arms use the *same* agent, so
  any gap is attributable to the meta-cognitive layer. Swap in `AnthropicLLM` for live runs.

## Layout

```
src/sakshi/
  core/   state, types, embeddings, llm (Mock/Sequenced/Cassette/Anthropic)
  tools/  registry, builtins (calc, docstore, untrusted web, crosscheck)
  agent/  metrics, observer, controller, loop, react        # Project 1
  bench/  tasks, workers, scenarios, harness, metrics, report # Project 2
  cos/    persistence, kernel                                # Project 3
  eval/   ablation + bootstrap CI     obs/ trace + manifest  cli.py
apps/dashboard/index.html             examples/  tests/
Makefile · Dockerfile · .github/workflows/ci.yml · ARCHITECTURE.md · PORTFOLIO.md
```

MIT licensed.
