"""Quantitative reliability signals computed from cognitive state.

These are deliberately interpretable, cheap to compute, and unit-testable. The
observer can *augment* them with LLM judgments, but the numeric backbone here
guarantees the controller always has a signal even with a mock model.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..core.embeddings import Embedder, semantic_distance
from ..core.state import BeliefState, CognitiveState

_NEG_RE = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|won't|doesn't|don't)\b")
_WORD_RE = re.compile(r"[a-z0-9]+")


def goal_drift(state: CognitiveState, embedder: Embedder) -> float:
    """Semantic distance between current focus and the original goal, in [0, 1]."""
    focus = state.goal.current_focus or state.goal.goal
    return semantic_distance(embedder, focus, state.goal.goal)


def _content_words(text: str) -> set:
    stop = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "for", "it"}
    return {w for w in _WORD_RE.findall(text.lower()) if w not in stop and len(w) > 2}


def belief_conflict(beliefs: BeliefState) -> float:
    """Heuristic contradiction score in [0, 1].

    Two beliefs conflict when they share most content words but differ in
    polarity (one negated, the other not). This catches the common
    "X is safe" / "X is not safe" class without an LLM, and the observer can
    override with a semantic judgment when a real model is present.
    """
    bs = [b for b in beliefs.beliefs if b.confidence > 0.1]
    if len(bs) < 2:
        return 0.0
    conflicts = 0
    pairs = 0
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            pairs += 1
            wi, wj = _content_words(bs[i].claim), _content_words(bs[j].claim)
            if not wi or not wj:
                continue
            overlap = len(wi & wj) / len(wi | wj)
            neg_i = bool(_NEG_RE.search(bs[i].claim.lower()))
            neg_j = bool(_NEG_RE.search(bs[j].claim.lower()))
            if overlap >= 0.5 and neg_i != neg_j:
                conflicts += 1
    return 0.0 if pairs == 0 else min(1.0, conflicts / pairs * 2.0)


def unverified_load(beliefs: BeliefState, threshold: float = 0.7) -> float:
    """Share of beliefs that are high-confidence yet unverified, in [0, 1]."""
    if not beliefs.beliefs:
        return 0.0
    risky = beliefs.unverified_high_conf(threshold)
    return len(risky) / len(beliefs.beliefs)


def aggregate_uncertainty(state: CognitiveState, drift: float, conflict: float) -> float:
    """Blend belief confidences with structural risk signals into one scalar."""
    bs = state.beliefs.beliefs
    if bs:
        mean_conf = sum(b.confidence for b in bs) / len(bs)
        belief_unc = 1.0 - mean_conf
    else:
        belief_unc = 0.5
    raw = 0.5 * belief_unc + 0.3 * conflict + 0.2 * drift
    return max(0.0, min(1.0, raw))


def compute_signals(state: CognitiveState, embedder: Embedder) -> Tuple[float, float, float, float]:
    """Return (drift, uncertainty, conflict, unverified_load)."""
    drift = goal_drift(state, embedder)
    conflict = belief_conflict(state.beliefs)
    unc = aggregate_uncertainty(state, drift, conflict)
    uvl = unverified_load(state.beliefs)
    return drift, unc, conflict, uvl
