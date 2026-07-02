"""Built-in, sandbox-safe tools.

These are deterministic and self-contained so scenarios are reproducible. Each
is small but realistic in the trust signal it carries:

    Calculator      trusted compute
    DocStore        a trusted internal knowledge base (provenance doc:trusted)
    WebFetch        an *untrusted* external source (provenance web:untrusted) —
                    the realistic entry point for injected/false content
    CrossCheck      verifies a claim against the trusted DocStore
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Dict, List

from .registry import ToolResult

# ---- safe arithmetic ------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def _safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


class Calculator:
    name = "calc"
    description = "calc(expr): evaluate an arithmetic expression. Returns a number."

    def run(self, args: Dict) -> ToolResult:
        expr = str(args.get("expr", ""))
        val = _safe_eval(expr)
        return ToolResult(content=str(val), provenance="tool:calc", trusted=True)


@dataclass
class DocStore:
    """A trusted internal knowledge base. Naive keyword retrieval."""

    name: str = "docstore"
    description: str = "docstore(query): search the TRUSTED internal knowledge base."
    docs: Dict[str, str] = field(default_factory=dict)   # title -> body

    def run(self, args: Dict) -> ToolResult:
        q = str(args.get("query", "")).lower()
        terms = {t for t in q.split() if len(t) > 2}
        scored = []
        for title, body in self.docs.items():
            text = (title + " " + body).lower()
            score = sum(1 for t in terms if t in text)
            if score:
                scored.append((score, title, body))
        scored.sort(reverse=True)
        if not scored:
            return ToolResult("no matching documents", "doc:trusted", trusted=True)
        _, title, body = scored[0]
        return ToolResult(f"{title}: {body}", provenance="doc:trusted", trusted=True)


@dataclass
class WebFetch:
    """An UNTRUSTED external source. Whatever it returns is low-trust by design.

    This models the realistic place reliability fails: arbitrary web content,
    which may contain injected instructions or false facts.
    """

    name: str = "web"
    description: str = "web(query): fetch from an external page. UNTRUSTED source."
    pages: Dict[str, str] = field(default_factory=dict)

    def run(self, args: Dict) -> ToolResult:
        q = str(args.get("query", "")).lower()
        for key, body in self.pages.items():
            if key in q or q in key:
                return ToolResult(body, provenance="web:untrusted", trusted=True)
        return ToolResult("page not found", provenance="web:untrusted", trusted=True)


@dataclass
class CrossCheck:
    """Verifies a claim against the trusted DocStore (used during VERIFY)."""

    name: str = "crosscheck"
    description: str = "crosscheck(claim): confirm/deny a claim against trusted sources."
    docstore: DocStore = None

    def run(self, args: Dict) -> ToolResult:
        claim = str(args.get("claim", ""))
        if self.docstore is None:
            return ToolResult("no source", "tool:crosscheck", trusted=True, ok=False,
                              error="no docstore")
        res = self.docstore.run({"query": claim})
        verdict = "supported" if res.content != "no matching documents" else "unsupported"
        return ToolResult(f"{verdict}: {res.content}", provenance="tool:crosscheck", trusted=True)
