"""
================================================================================
 USAGE TRACKER  —  every OpenAI call, its tokens and its (estimated) cost
================================================================================

RAG costs money in two places:
  1. Embeddings  — once per chunk at index time (cached!) + once per question.
  2. Chat        — every question (rewriting, reranking, final answer).

Every API call is recorded here; type `/stats` in the chat to see the totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import PRICES_PER_1M_TOKENS


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    prices = PRICES_PER_1M_TOKENS.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def format_cost(cost: float | None) -> str:
    return "n/a (model not in price table)" if cost is None else f"${cost:.6f}"


@dataclass
class APICall:
    purpose: str          # e.g. "embed chunks", "rewrite query", "answer"
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float

    @property
    def cost(self) -> float | None:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


@dataclass
class UsageTracker:
    calls: list[APICall] = field(default_factory=list)

    def record(self, purpose: str, model: str, input_tokens: int,
               output_tokens: int, seconds: float) -> APICall:
        call = APICall(purpose, model, input_tokens, output_tokens, seconds)
        self.calls.append(call)
        return call

    @property
    def total_cost(self) -> float:
        return sum(c.cost or 0.0 for c in self.calls)


# Shared, session-wide tracker.
usage = UsageTracker()
