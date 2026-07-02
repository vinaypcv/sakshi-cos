# Runbook — Live Scale Evaluation

Everything below runs offline today (deterministic mock validates the whole
pipeline). To produce *real* numbers you add an API key and the `--live` flag.
This is now a data-collection exercise, not an engineering one.

## 0. Prerequisites

```bash
pip install -e ".[dev,llm]"      # installs anthropic + pytest
export ANTHROPIC_API_KEY=sk-...  # your key
export PYTHONPATH=src
```

## 1. Pre-flight: estimate the bill before spending

```bash
python -m sakshi scale --dry-run --n 50 --model claude-sonnet-4-6
```

Refine the estimate with a tiny live pilot, read actual tokens/task from the
cost table, then re-estimate:

```bash
python -m sakshi --out runs/pilot scale --live --n 2 --cassette runs/pilot.cassette.json
```

## 2. Record the run (cassette = reproducibility)

```bash
python -m sakshi --out runs/scale scale --live --n 50 \
    --model claude-sonnet-4-6 \
    --cassette runs/scale.cassette.json --cassette-mode record
```

This runs 50 tasks/mode x 4 modes x 2 arms against the real model, writes
`runs/scale/scale_report.{md,json}`, and saves every model response to the
cassette. The report has per-mode success with 95% Wilson CIs, the paired
baseline->Sakshi delta with a CI, cost/latency per arm, signal calibration
(AUC), and intervention precision/recall.

## 3. Replay forever, free, in CI

```bash
python -m sakshi --out runs/scale scale --live --n 50 \
    --cassette runs/scale.cassette.json --cassette-mode replay
```

Identical numbers, no key, no spend. Commit the cassette; CI replays it.

## 4. Compare models (cheap, high-signal)

```bash
for m in claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-8; do
  python -m sakshi --out runs/$m scale --live --n 50 --model $m \
      --cassette runs/$m.cassette.json
done
```

Does the meta-cognitive layer help more on the smaller, cheaper model? (Usually
yes — that is a compelling cost-reliability story.)

## 5. Scale up

`--n 50` is the floor for tight CIs. For a flagship result use `--n 100+` and
multiple seeds (`--seed`), then average. Above ~30 model calls per task,
parallelize: shard tasks across processes (each writes its own cassette) and
merge the JSON reports.

---

# What turns this from "good" into "exceptional"

Built in already (use them — they are what reviewers look for):

- **Procedural tasks with known ground truth** — 200+ distinct, auto-graded tasks, not a
  hand-curated toy set.
- **Paired statistics** — same tasks both arms; Wilson CIs + paired-difference CI, so the
  improvement claim is defensible, not a single number.
- **Ablations** — each control action removed drops exactly the failure it targets.
- **Detector calibration** — treat the observer's signals as a *classifier* and report AUC for
  predicting failure. This reframes the project as measurement science, not just engineering.
  (Offline AUC is near chance by construction — the mock's outcome is drawn from a difficulty
  model independent of the realized signal; on live data this number becomes meaningful and is
  the single most impressive plot in the writeup.)
- **Cost/latency accounting** — the report quantifies the tokens/$ overhead of reliability.

Add these to stand out further:

1. **Cost-reliability Pareto curve** — sweep controller thresholds; plot success vs $/task.
   The story "reliability is a dial you can price" is rare and senior.
2. **Real adversarial corpora** — replace generated injections with captured prompt-injection
   strings and genuinely contradictory web sources; add an LLM-judge grader for free-form
   answers (`AnthropicLLM` + a grading prompt) alongside the rule grader.
3. **Held-out generalization** — calibrate thresholds on one seed's tasks, report on another.
   Show the policy is not overfit.
4. **A short writeup + the telemetry dashboard + a 2-minute demo video.** Load
   `runs/scale/...` traces into `apps/dashboard/index.html`. The narrative ("I found a failure
   mode, defined metrics, built a detector, proved it with CIs and ablations, and priced it")
   is the differentiator — most portfolios show an app, not a result.
5. **One human-readable failure analysis** — pull 3 tasks where Sakshi still failed and explain
   why. Owning the limitations reads as seniority.

## Honesty checklist (say these out loud; they signal maturity)

- Offline numbers validate the *pipeline*, not the model; live `--live` numbers are the result.
- The default drift detector is lexical; semantic drift wants a learned embedder.
- Pricing constants in `eval/cost.py` are illustrative — set current rates before quoting `$`.
- Procedurally generated tasks are easier than messy real ones; report both if you can.
