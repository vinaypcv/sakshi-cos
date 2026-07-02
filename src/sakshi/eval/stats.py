"""Statistics for honest reporting at scale.

- Wilson score interval: the right CI for a success *proportion* (better than
  normal-approx at the rates and sample sizes we care about).
- Paired difference CI: baseline and Sakshi run the SAME tasks, so the comparison
  is paired; we report the mean per-task difference with a CI.
- AUC (Mann-Whitney): detector quality of a continuous signal (e.g. goal_drift)
  for predicting task failure.
"""
from __future__ import annotations

import math
from typing import List, Tuple

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> Tuple[float, float, float]:
    """Return (point, lo, hi) for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def paired_diff_ci(a: List[float], b: List[float], z: float = Z95) -> Tuple[float, float, float]:
    """Mean of (a_i - b_i) with a normal CI. a, b are aligned per-task outcomes."""
    if len(a) != len(b) or not a:
        return 0.0, 0.0, 0.0
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    if n < 2:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - z * se, mean + z * se


def auc(scores: List[float], labels: List[int]) -> float:
    """Area under ROC via the Mann-Whitney U statistic.

    labels: 1 = positive (e.g. task failed), 0 = negative. Returns P(score of a
    random positive > score of a random negative), with ties counted as 0.5.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for sp in pos:
        for sn in neg:
            wins += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return wins / (len(pos) * len(neg))
