# Sakshi-COS Technical Report

## Abstract

Sakshi-COS is a meta-cognitive reliability layer for long-horizon AI agents. It separates task execution from reliability monitoring by wrapping a tool-using worker in an observer-controller loop. The system tracks persistent cognitive state, detects goal drift, memory pollution, overconfidence, and belief conflict, then intervenes through verify, reflect, and replan directives.

## Problem

Long-horizon agents often fail because they lose stable state across many actions. They can drift from the original goal, trust unverified retrieval output, accumulate contradictory beliefs, or answer confidently before verification.

## Method

Sakshi-COS represents state as goal, current focus, beliefs, memory, uncertainty, and history. A meta-cognitive observer computes drift, conflict, unverified-load, and uncertainty signals. A reliability controller maps those signals to explicit interventions.

## Evaluation

The evaluation compares a passive baseline against the Sakshi-controlled agent on the same tasks. The benchmark includes scripted failure modes, tool-using ReAct scenarios, prompt-injection retrieval cases, ablations, and scale evaluations with confidence intervals.

## Reproducibility

Run:

```bash
python -m pytest -q
python -m sakshi --out runs all
python -m sakshi --out runs_scale scale --n 50
```

Live evaluation requires an Anthropic API key:

```bash
python -m sakshi --out runs_live scale --n 5 --live --model claude-sonnet-5 --cassette cassettes/sonnet5-pilot.json --cassette-mode record
```

## Limitations

Offline results are not a substitute for live-model evaluation. Generated tasks are useful for controlled measurement but should be complemented with real adversarial retrieval datasets and externally reviewed test cases.
