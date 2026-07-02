"""Lightweight observability.

A run emits a JSONL trace (one span per agent step) plus a manifest capturing
config, environment, and aggregate metrics. No external dependency; the JSONL is
trivially ingestible by jq, pandas, or an OTel collector later.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..core.state import CognitiveState


@dataclass
class RunRecorder:
    out_dir: str
    run_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self.trace_path = os.path.join(self.out_dir, f"{self.run_id}.trace.jsonl")
        self.manifest_path = os.path.join(self.out_dir, f"{self.run_id}.manifest.json")
        open(self.trace_path, "w").close()

    def log_steps(self, state: CognitiveState) -> None:
        with open(self.trace_path, "a") as f:
            for rec in state.history:
                f.write(json.dumps({"run_id": self.run_id, "session": state.session_id,
                                    **rec.to_dict()}) + "\n")

    def manifest(self, metrics: Dict[str, Any], extra: Optional[Dict] = None) -> str:
        data = {
            "run_id": self.run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": round(time.time() - self._t0, 3),
            "python": platform.python_version(),
            "config": self.config,
            "metrics": metrics,
            **(extra or {}),
        }
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        return self.manifest_path
