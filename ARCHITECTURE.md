# Architecture

Sakshi-COS is one package, `sakshi`, organized as five layers. Dependencies point
downward only; nothing in `core` imports from `agent`, `bench`, `cos`, etc.

```
                 ┌─────────────────────────────────────────────┐
   cos  (kernel) │ SakshiKernel · CognitiveStore (persist/resume)│   Project 3
                 └───────────────┬─────────────────────────────┘
                                 │ owns + wires
 agent (Project 1)   ┌───────────▼───────────┐   bench (Project 2)   eval / obs
 observer ─ metrics  │  SakshiAgent loop      │   tasks · workers     ablation
 controller          │  Worker / ReAct worker │   scenarios · harness traces
                 ┌───┴───────────────────────┴───┐ report · metrics
 tools           │  ToolRegistry · builtins      │
                 └───────────────┬───────────────┘
 core            CognitiveState · types · embeddings · LLM (Mock/Sequenced/Cassette/Anthropic)
```

## The control loop

```
worker.step(state)            → WorkerStep (output, focus, beliefs, memory, done)
  apply to CognitiveState     → goal.current_focus, beliefs, memory (with provenance)
observer.assess(state)        → Assessment{goal_drift, uncertainty, conflict, unverified_load, flagged}
controller.decide(assessment) → CONTINUE | VERIFY | REFLECT | REPLAN   (priority + cooldown)
worker.on_directive(action)   → structural correction (refocus, distrust memory, retire claims)
kernel.store.save(state)      → checkpoint; resumable next session
```

The observer judges from cognitive state, never from the worker's private chain of
thought — that separation is what makes it meta-cognition rather than self-grading.

## Signals (all in [0, 1], all interpretable)

- **goal_drift** — semantic distance between `current_focus` and the original goal
  (`Embedder`; default is an offline stable hashing embedder, swappable for a learned model).
- **uncertainty** — blend of mean belief confidence with drift and conflict.
- **conflict_score** — contradiction among *active* beliefs (shared content words, opposite polarity).
- **unverified_load** — share of high-confidence beliefs that remain unverified.

## Controller policy

Priority `REPLAN > REFLECT > VERIFY > CONTINUE`, each behind a tunable threshold and an
enable flag (used by the ablation), with a cooldown so an intervention does not thrash.
Every decision is explainable from the thresholds — important for an evaluation system.

## Trust & provenance

Every `MemoryItem` and `ToolResult` carries a `provenance` (`doc:trusted`, `web:untrusted`,
`tool:calc`, …) and a `trusted` flag. Memory pollution is detected as a *trusted item from a
suspicious provenance*, which models the realistic failure: an agent believing an untrusted
web page. VERIFY distrusts those items and retires the claims that rested on them.

## LLM boundary

Everything talks to the `LLM` protocol:
- `MockLLM` / `SequencedLLM` — deterministic, for tests and offline demos (exercise the *real*
  orchestration; only the model tokens are canned — standard practice for testing LLM apps).
- `CassetteLLM` — record once against a live model, replay forever in CI.
- `AnthropicLLM` — live inference (`pip install anthropic`, `ANTHROPIC_API_KEY`).

## Extension points

| Want to… | Implement / swap |
| --- | --- |
| Use semantic drift | a learned `Embedder` |
| Add a tool | a `Tool` (name, description, `run`) → `ToolRegistry.register` |
| Add a failure mode | a `Task` (scripted) or a `Scenario` (tool-using) + grader |
| Change intervention policy | `ControllerConfig` thresholds / enable flags |
| Run a real agent | `LLMReActWorker(llm=AnthropicLLM(...), tools=...)` |
| Persist elsewhere | replace `CognitiveStore` (JSON today; SQLite/Redis later) |
```
