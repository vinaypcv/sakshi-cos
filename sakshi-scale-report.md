# Sakshi-COS Scale Report

200 tasks (50/mode x 4 modes), model `claude-sonnet-4-6`. Arms run the SAME tasks (paired).

## Success by failure mode (95% Wilson CI)

| Mode | Baseline | Sakshi | Paired Δ (95% CI) |
| --- | ---: | ---: | ---: |
| goal_drift | 0.10 [0.04,0.21] | 0.62 [0.48,0.74] | +0.52 [+0.37,+0.67] |
| memory_pollution | 0.08 [0.03,0.19] | 0.74 [0.60,0.84] | +0.66 [+0.53,+0.79] |
| overconfidence | 0.26 [0.16,0.40] | 0.66 [0.52,0.78] | +0.40 [+0.23,+0.57] |
| belief_conflict | 0.16 [0.08,0.29] | 0.74 [0.60,0.84] | +0.58 [+0.43,+0.73] |
| **overall** | **0.15** | **0.69** | **+0.54 [+0.46,+0.62]** |

## Cost & latency

| Arm | Calls | In tok | Out tok | USD | USD/task | s/task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 400 | 72861 | 7221 | 0.3269 | 0.00163 | 0.006 |
| sakshi | 574 | 121561 | 9867 | 0.5127 | 0.00256 | 0.009 |

_Pricing is configurable; verify current rates. Offline tokens are estimated._

## Detector quality (signal -> failure, AUC)

| Signal | AUC |
| --- | ---: |
| goal_drift | 0.5166 |
| uncertainty | 0.5097 |
| conflict | 0.5 |
| unverified_load | 0.451 |

AUC = P(signal higher on a failed task than a passed one); 0.5 = no signal.

## Intervention targeting

Precision 0.85, recall 1.00 (200 intervened, 170 would-fail).
