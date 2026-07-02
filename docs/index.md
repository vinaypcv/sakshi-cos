# Sakshi-COS: Meta-Cognitive Reliability for Long-Horizon AI Agents

Sakshi-COS is a reliability layer for long-horizon AI agents. It maintains a persistent cognitive state and uses a separate observer-controller loop to detect goal drift, belief conflict, unverified memory, and overconfidence.

## Core idea

Most agent stacks look like this:

```text
context + retrieval + prompt -> next action
```

Sakshi-COS adds persistent state and meta-cognitive control:

```text
persistent goal + beliefs + memory + observer + controller -> safer next action
```

## Architecture

```text
Worker / ReAct Agent
        |
        v
Cognitive State: goal, focus, beliefs, memory, uncertainty, history
        |
        v
Meta-Cognitive Observer: drift, uncertainty, conflict, unverified load
        |
        v
Reliability Controller: CONTINUE, VERIFY, REFLECT, REPLAN
        |
        v
Corrective directive back to worker
```

## Evidence generated so far

The repository includes benchmark reports under `docs/results/`.

Local validation:

```text
python -m pytest -q --basetemp=.pytest_tmp
python -m sakshi --out runs all
python -m sakshi --out runs_scale scale --n 50
```

## Evaluation modes

| Mode | Purpose |
| --- | --- |
| `bench` | Small scripted benchmark for core failure modes |
| `scenarios` | Tool-using ReAct scenarios with retrieval and prompt-injection risk |
| `ablate` | Removes controller actions to prove each action matters |
| `scale` | 50–100 generated tasks per failure mode with confidence intervals |
| `scale --live` | Same benchmark against a real Claude model |

## Claims and limitations

The offline benchmark validates the architecture, instrumentation, and deterministic evaluation harness. Live-model results should be reported separately with model name, date, cost, sample size, and replay cassette hash. The project should not claim broad production reliability until evaluated on larger live and adversarial corpora.
