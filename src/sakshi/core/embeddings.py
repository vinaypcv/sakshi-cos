"""Embeddings used for goal-drift measurement.

Goal drift is measured as the semantic distance between the agent's *current
focus* and its *original goal*. We need an embedding for that. To keep the whole
system runnable with zero network access and fully deterministic (important for
a reproducible benchmark), the default embedder is a local hashing embedder over
character n-grams. A real embedder (Anthropic, OpenAI, sentence-transformers,
etc.) can be dropped in by implementing the `Embedder` protocol.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_hash(text: str) -> int:
    """Process-independent hash so embeddings (and the benchmark) reproduce."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """Deterministic bag-of-n-grams embedding hashed into a fixed-size vector.

    Not as expressive as a learned model, but it is stable, fast, offline, and
    captures enough lexical/semantic overlap to drive a meaningful drift signal
    for evaluation. Swappable for a real model in production.
    """

    def __init__(self, dim: int = 256, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def _tokens(self, text: str) -> List[str]:
        words = _TOKEN_RE.findall(text.lower())
        toks: List[str] = list(words)
        joined = " ".join(words)
        for i in range(len(joined) - self.ngram + 1):
            toks.append(joined[i : i + self.ngram])
        return toks

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = _stable_hash(f"{self.ngram}:{tok}") % self.dim
            sign = 1.0 if (_stable_hash(tok) >> 7) & 1 else -1.0
            vec[h] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def semantic_distance(embedder: Embedder, a: str, b: str) -> float:
    """Return a drift value in [0, 1]; 0 = identical focus, 1 = unrelated."""
    sim = cosine(embedder.embed(a), embedder.embed(b))
    # cosine here is in [-1, 1] but for hashed bags is effectively [0, 1].
    sim = max(0.0, min(1.0, sim))
    return 1.0 - sim
