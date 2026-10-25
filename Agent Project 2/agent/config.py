"""
================================================================================
 CONFIGURATION  —  every "knob" of the agent lives in this one file
================================================================================

In Project 1 (RAG) the knobs were about SEARCH: chunk size, top_k, BM25, ...
In an AGENT the knobs are about CONTROL: how many steps the model may take,
how much money it may spend, which tools it may use, whether a human must
approve risky actions, and which architecture runs the show:

    react  ->  one agent in a Think -> Act -> Observe loop        (core/react_agent.py)
    plan   ->  first write a plan, then execute it step by step   (core/planner.py)
    multi  ->  a supervisor delegates to specialist agents        (core/supervisor.py)

Change any value while the bot is running with:   /set <key> <value>
then ask the same question again and compare the logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# The project root is the folder that contains `main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Read-only company data the tools look things up in.
DATA_DIR = PROJECT_ROOT / "data"
# The ONLY folder the agent's file tools may read from / write to.
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
# Things the agent changes at runtime: tickets, long-term memory.
STATE_DIR = PROJECT_ROOT / ".agent_state"
# One JSON trace per question, so you can replay what the agent did.
TRACES_DIR = PROJECT_ROOT / "traces"

# Read OPENAI_API_KEY (and optional model override) from the .env file.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class AgentConfig:
    # ------------------------------------------------------------------ #
    # 1. MODEL
    # ------------------------------------------------------------------ #
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    # Low temperature = the agent picks tools consistently between runs.
    temperature: float = 0.2
    # Upper bound on the length of ONE model reply (a thought + tool calls,
    # or the final answer).
    max_output_tokens: int = 800

    # ------------------------------------------------------------------ #
    # 2. ARCHITECTURE
    # ------------------------------------------------------------------ #
    # "react": a single agent loops Think -> Act -> Observe until it is done.
    # "plan":  a planner writes a numbered plan first, an executor runs each
    #          step, and a final call writes the answer from the results.
    # "multi": a supervisor agent delegates sub-tasks to specialist agents
    #          (researcher, analyst, operator), each with its own tools.
    agent_mode: str = "react"

    # ------------------------------------------------------------------ #
    # 3. LOOP CONTROL + GUARDRAILS  (what stops an agent running away)
    # ------------------------------------------------------------------ #
    # Max Think->Act->Observe rounds for ONE agent run. When it is reached the
    # agent is forced to answer with what it has.
    max_iterations: int = 8
    # Max estimated spend (USD) for ONE question, across every agent and call.
    max_cost_usd: float = 0.05
    # Let the model request several tool calls in one reply (e.g. look up
    # three products at once) instead of one round trip per tool.
    parallel_tool_calls: bool = True
    # Human-in-the-loop: ask YOU before running a tool that changes something
    # (create a ticket, write a file). Read-only tools never ask.
    require_approval: bool = True
    # Tool results longer than this are cut before going back to the model.
    # Long observations make every following step more expensive.
    max_tool_output_chars: int = 3000
    # Comma-separated tool names the agent is NOT allowed to see.
    # Easiest way to change: /disable <tool>  and  /enable <tool>.
    disabled_tools: str = ""

    # ------------------------------------------------------------------ #
    # 4. REASONING
    # ------------------------------------------------------------------ #
    # Ask the model to write one sentence of reasoning ("thought") before each
    # tool call. Costs a few tokens, makes the agent far easier to follow.
    show_thoughts: bool = True
    # After the answer, a separate "critic" call checks it against the tool
    # results. If it finds problems, the agent gets one more chance.
    use_reflection: bool = False
    # How many critic -> revise rounds are allowed.
    max_reflection_rounds: int = 1

    # ------------------------------------------------------------------ #
    # 5. MEMORY
    # ------------------------------------------------------------------ #
    # Short-term: how many previous Q/A turns are re-sent with each question.
    history_turns: int = 4
    # Long-term: give the agent `remember` / `recall` tools and put the saved
    # facts in its system prompt. Stored in .agent_state/long_term_memory.json
    # so they survive a restart.
    use_long_term_memory: bool = True

    # ------------------------------------------------------------------ #
    # 6. LOGGING
    # ------------------------------------------------------------------ #
    # 0 = answers only | 1 = steps + tool calls | 2 = + tables & WHY notes
    # 3 = + everything (full messages, tool schemas, raw tool output)
    verbosity: int = 3
    # Also write a full (verbosity 3) log of the session to logs/.
    log_to_file: bool = True
    # Save a JSON trace of every question to traces/.
    save_traces: bool = True

    # ------------------------------------------------------------------ #
    # Helpers used by the /set and /config commands
    # ------------------------------------------------------------------ #
    def set_value(self, key: str, raw_value: str) -> object:
        """Parse `raw_value` to the right type for `key`, validate and assign it."""
        if key not in {f.name for f in fields(self)}:
            raise KeyError(f"Unknown setting '{key}'. Type /config to see all settings.")

        current = getattr(self, key)
        if isinstance(current, bool):
            lowered = raw_value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                value: object = True
            elif lowered in ("false", "0", "no", "off"):
                value = False
            else:
                raise ValueError(f"'{key}' expects true/false, got '{raw_value}'")
        elif isinstance(current, int):
            value = int(raw_value)
        elif isinstance(current, float):
            value = float(raw_value)
        else:
            value = raw_value.strip()

        if key in CHOICES and value not in CHOICES[key]:
            raise ValueError(f"'{key}' must be one of {CHOICES[key]}")
        if key in ("max_iterations", "max_output_tokens", "max_tool_output_chars") and value < 1:
            raise ValueError(f"'{key}' must be at least 1")

        setattr(self, key, value)
        return value

    def as_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def disabled(self) -> set[str]:
        return {t.strip() for t in self.disabled_tools.split(",") if t.strip()}


# Allowed values for "enum-like" settings.
CHOICES: dict[str, tuple[object, ...]] = {
    "agent_mode": ("react", "plan", "multi"),
    "verbosity": (0, 1, 2, 3),
}

# Approximate prices in USD per 1 MILLION tokens: (input, output).
# Used only to print cost estimates in the logs and to enforce max_cost_usd —
# check openai.com/pricing for the current numbers. Unknown models are shown
# as "n/a" and count as $0 towards the budget.
PRICES_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}
