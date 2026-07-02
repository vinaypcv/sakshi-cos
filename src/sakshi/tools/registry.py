"""Tool layer.

Real agents act through tools, and tool *outputs* are where most reliability
failures enter: an untrusted web page injects a false fact, two sources
disagree, a calculation is wrong. Every tool result carries a `provenance` and a
`trusted` flag so the observer can reason about where a belief came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol


@dataclass
class ToolResult:
    content: str
    provenance: str            # e.g. "tool:calc", "web:untrusted", "doc:trusted"
    trusted: bool = True
    ok: bool = True
    error: str = ""

    def render(self) -> str:
        if not self.ok:
            return f"[error: {self.error}]"
        return self.content


class Tool(Protocol):
    name: str
    description: str

    def run(self, args: Dict) -> ToolResult:
        ...


@dataclass
class ToolRegistry:
    tools: Dict[str, Tool] = field(default_factory=dict)
    call_log: List[Dict] = field(default_factory=list)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def spec(self) -> str:
        """Human/LLM-readable catalogue for the agent prompt."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def call(self, name: str, args: Dict) -> ToolResult:
        if name not in self.tools:
            res = ToolResult("", "tool:unknown", trusted=False, ok=False,
                             error=f"no such tool '{name}'")
        else:
            try:
                res = self.tools[name].run(args)
            except Exception as exc:  # tools must never crash the loop
                res = ToolResult("", f"tool:{name}", trusted=False, ok=False, error=str(exc))
        self.call_log.append({"name": name, "args": args, "ok": res.ok,
                              "provenance": res.provenance})
        return res
