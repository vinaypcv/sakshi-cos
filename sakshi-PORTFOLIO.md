# Portfolio Completion Guide

This turns the three-project plan into a finished, defensible portfolio piece. It
states honestly what is built, what each part proves, what to say in interviews,
and the remaining work that moves it from "strong prototype" to "research result."

## Completion checklist

**Project 1 — Sakshi Agent (reliability layer)** — done
- [x] Meta-cognitive observer judging from cognitive state, not the worker's reasoning
- [x] Interpretable metrics: goal drift, uncertainty, belief conflict, unverified load
- [x] Controller with explicit priority policy, thresholds, enable flags, cooldown
- [x] Agent loop wrapping any `Worker` (scripted stub, mock, or real LLM)

**Project 2 — SakshiBench (evaluation)** — done
- [x] Scripted suite inducing 4 failure modes (drift, pollution, overconfidence, conflict)
- [x] Tool-using ReAct scenarios over a corpus (research-drift, web-injection)
- [x] Fair baseline-vs-Sakshi protocol (same agent; controller is the only variable)
- [x] Run metrics + JSON traces + markdown report
- [x] Ablation isolating each control action; bootstrap CIs

**Project 3 — Sakshi-COS (integration)** — done
- [x] Kernel integrating observer + controller over persistent cognitive state
- [x] Checkpoint + resume from disk (cognitive state survives sessions)
- [x] Unified status API

**Engineering (what makes it a portfolio, not a notebook)** — done
- [x] Tool layer with provenance/trust; safe built-in tools
- [x] LLM boundary: Mock / Sequenced / Cassette (record-replay) / Anthropic
- [x] Observability: JSONL step traces + run manifests
- [x] CLI (`python -m sakshi …`), Makefile, GitHub Actions CI (3 Python versions), Dockerfile
- [x] 18 tests, deterministic; telemetry dashboard
- [x] README, ARCHITECTURE, this guide, MIT license

## Headline numbers (deterministic, reproducible)

| Suite | Baseline | Sakshi |
| --- | ---: | ---: |
| Scripted (4 modes) | 0.25 | 1.00 |
| ReAct scenarios (2) | 0.00 | 1.00 |
| Ablation: each action removed | drops the task it targets | full = 1.00, none = 0.25 |

`make all` regenerates every number and artifact.

## What to say in an interview

> "Long-horizon agents fail by drifting off-goal, trusting polluted memory, and skipping
> verification. I built a meta-cognitive layer: a separate observer quantifies those risks
> from the agent's cognitive state, and a controller decides whether to continue, verify,
> reflect, or replan over a persistent state that survives sessions. I built a benchmark that
> induces each failure mode and an ablation showing each control action recovers exactly the
> failure it targets — same agent, controller toggled, so the gain is attributable to the
> layer. It runs deterministically offline for CI and against a live model with one flag."

## Resume bullet

> Built **Sakshi-COS**, a meta-cognitive reliability layer for long-horizon LLM agents:
> goal-drift / belief-conflict / verification metrics driving a continue-verify-reflect-replan
> controller over persistent cognitive state; a failure-mode benchmark with ablations
> (baseline 0.25 → 1.00 success), a tool-using ReAct agent, record-replay LLM testing, CI, and
> a telemetry dashboard.

## Honesty notes (say these; they signal seniority)

- The benchmark's base agents are deterministic *simulators* of each failure mode, and the
  offline scenario model is a prompt-sensitive mock. This is deliberate: it makes the
  baseline-vs-Sakshi comparison reproducible and isolates the controller's effect. The
  *same* real `LLMReActWorker` runs live with `AnthropicLLM`.
- The default drift embedder is lexical; semantic drift needs a learned embedder.
- Task counts are small (n=4, n=2); the bootstrap CIs are wide. Scale the suites before
  making strong quantitative claims.

## Next work (highest-value first)

1. **Live runs at scale:** record cassettes for 50+ tasks per failure mode with a real model;
   report success with tighter CIs and cost/latency.
2. **Real adversarial tasks:** prompt-injection corpora, distractor documents, stale-memory
   multi-session tasks; LLM-judge graders for free-form answers.
3. **Learned embedder** for semantic drift; per-task threshold calibration.
4. **Cost/reliability curves:** interventions add tokens — quantify the trade-off.
5. **SQLite persistence**, multi-session goal hierarchies, observer-as-LLM folded into metrics.
