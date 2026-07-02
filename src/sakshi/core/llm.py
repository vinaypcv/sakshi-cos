"""LLM abstraction.

Everything in Sakshi talks to an `LLM` protocol, never to a vendor SDK directly.
This makes the whole system runnable offline/deterministically via `MockLLM`
(used by the benchmark and tests) while allowing a real model to be plugged in
for live agents via `AnthropicLLM`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_s += other.latency_s
        self.calls += other.calls

    def to_dict(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "latency_s": round(self.latency_s, 3), "calls": self.calls}


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        ...


class MockLLM:
    """Deterministic stand-in.

    Returns canned-but-stable responses derived from the prompt hash. It is good
    enough to exercise control flow, observer prompts, and reflection steps
    without a network call. Optionally seeded with scripted responses keyed by a
    substring match for richer demos.
    """

    def __init__(self, scripted: Optional[dict] = None) -> None:
        self.scripted = scripted or {}
        self.calls: List[str] = []

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        for key, resp in self.scripted.items():
            if key in prompt:
                return resp
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=4).hexdigest()
        return f"[mock-response::{digest}]"


class SequencedLLM:
    """Returns a fixed queue of responses in order, ignoring the prompt.

    Used to drive the *real* ReAct agent code path deterministically in tests:
    the orchestration (tool dispatch, state updates, observer, controller) is
    exercised for real; only the model's tokens are canned. This is the standard
    way to test LLM applications without paying for or depending on inference.
    """

    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.i = 0
        self.calls: List[str] = []

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        if self.i >= len(self.responses):
            return "FINAL: (sequence exhausted)"
        resp = self.responses[self.i]
        self.i += 1
        return resp


class CassetteLLM:
    """Record/replay wrapper for reproducible runs with a real model.

    - mode="record": calls `inner` (a live LLM), saves prompt-hash -> response.
    - mode="replay": serves saved responses; raises if a prompt is unseen.

    Capture once with a key, then run the same evaluation forever offline and in
    CI with identical model behavior. Cassettes are JSON on disk.
    """

    def __init__(self, path: str, mode: str = "replay", inner: Optional[LLM] = None) -> None:
        self.path = path
        self.mode = mode
        self.inner = inner
        self._cache: dict = {}
        if os.path.exists(path):
            with open(path) as f:
                self._cache = json.load(f)

    @staticmethod
    def _key(prompt: str, system: Optional[str]) -> str:
        payload = ((system or "") + "\x1f" + prompt).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        k = self._key(prompt, system)
        if self.mode == "replay":
            if k not in self._cache:
                raise KeyError(f"cassette miss in replay mode ({self.path}); "
                               "re-record with a live model")
            return self._cache[k]
        if self.inner is None:  # pragma: no cover
            raise ValueError("record mode requires an inner LLM")
        resp = self.inner.complete(prompt, system=system, max_tokens=max_tokens,
                                   temperature=temperature)  # pragma: no cover
        self._cache[k] = resp
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._cache, f, indent=2)
        return resp


class AnthropicLLM:
    """Thin adapter over the Anthropic Messages API.

    Requires `anthropic` installed and ANTHROPIC_API_KEY set. Import is lazy so
    the rest of the system never hard-depends on the SDK.
    """

    def __init__(self, model: str = "claude-sonnet-5",
                 api_key: Optional[str] = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "AnthropicLLM requires the 'anthropic' package: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.last_usage = Usage()

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:  # pragma: no cover
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a precise reasoning component inside an AI agent.",
            messages=[{"role": "user", "content": prompt}],
        )
        u = getattr(msg, "usage", None)
        self.last_usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) if u else 0,
            output_tokens=getattr(u, "output_tokens", 0) if u else 0,
            calls=1,
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)   # ~4 chars/token heuristic when no real count


class MeteredLLM:
    """Wraps any LLM, recording per-call token usage and latency into a ledger.

    Uses the inner model's real usage when available (AnthropicLLM.last_usage);
    otherwise estimates tokens from text length. `sim_latency` adds a synthetic
    delay so offline cost/latency reporting is exercised realistically.
    """

    def __init__(self, inner: LLM, sim_latency: float = 0.0) -> None:
        self.inner = inner
        self.sim_latency = sim_latency
        self.total = Usage()
        self.ledger: List[Usage] = []

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        t0 = time.time()
        out = self.inner.complete(prompt, system=system, max_tokens=max_tokens,
                                  temperature=temperature)
        latency = (time.time() - t0) + self.sim_latency
        u = getattr(self.inner, "last_usage", None)
        if u is None or (u.input_tokens == 0 and u.output_tokens == 0):
            u = Usage(input_tokens=_estimate_tokens((system or "") + prompt),
                      output_tokens=_estimate_tokens(out), calls=1)
        u.latency_s = latency
        self.ledger.append(u)
        self.total.add(u)
        return out
