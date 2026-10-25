"""
================================================================================
 USAGE TRACKER  —  every OpenAI call, its tokens and its (estimated) cost
================================================================================

Agents are much more expensive than a single chat call, for two reasons:
  1. One question = MANY calls (one per Think step, plus planner/critic/...).
  2. Every call re-sends the WHOLE conversation so far: the system prompt, the
     tool schemas, every earlier tool call and every tool result. The prompt
     grows with each step.

Every API call is recorded here; type `/stats` to see them all. The Budget
class turns `max_cost_usd` into a hard stop for one question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.config import PRICES_PER_1M_TOKENS


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    prices = PRICES_PER_1M_TOKENS.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def format_cost(cost: float | None) -> str:
    return "n/a (model not in price table)" if cost is None else f"${cost:.6f}"


@dataclass
class APICall:
    purpose: str          # e.g. "react step 2", "planner", "critic"
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


class Budget:
    """The spending limit for ONE question (all agents and calls together)."""

    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = max_cost_usd
        self.start_cost = usage.total_cost
        self.start_call = len(usage.calls)

    @property
    def spent(self) -> float:
        return usage.total_cost - self.start_cost

    @property
    def calls(self) -> list[APICall]:
        return usage.calls[self.start_call:]

    def exceeded(self) -> bool:
        return self.spent >= self.max_cost_usd
