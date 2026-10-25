"""
================================================================================
 GUARDRAILS  —  human approval for actions that change things
================================================================================

An agent loop has three classic guardrails. Two of them live in the loop
(core/react_agent.py):
  1. max_iterations   - stop a model that keeps calling tools forever
  2. max_cost_usd     - stop a run that is getting too expensive
and the third lives here:
  3. HUMAN APPROVAL   - before a tool with side effects runs, a person sees
                        the exact call and decides.

Answering "no" does NOT crash anything. The refusal (and your reason, if you
type one) goes back to the model as the tool's result, and the model has to
deal with it, just like any other observation. Try denying a ticket and watch
what the agent does next.
"""

from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.logger import esc, log
from agent.tools.registry import Tool


class ApprovalGate:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        # Tools the user answered "always" for, for the rest of this session.
        self.always_allowed: set[str] = set()

    def check(self, tool: Tool, raw_arguments: str) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        if not tool.side_effect:
            return True, "read-only tool"
        if not self.cfg.require_approval:
            return True, "require_approval is off"
        if tool.name in self.always_allowed:
            return True, "always allowed this session"

        try:
            pretty = json.dumps(json.loads(raw_arguments), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = str(raw_arguments)
        log.block(f"APPROVAL NEEDED: the agent wants to run '{tool.name}'", pretty, level=0, style="yellow")
        log.explain("This tool changes something, so a human decides. A 'no' is sent back to the model as the tool result.")
        answer = log.console.input(
            "  [bold yellow]Allow?[/] [y]es / [n]o / [a]lways for this tool / or type a reason to deny: "
        ).strip()
        log.always(f"  [grey62](you answered: {esc(answer or 'n')})[/]")

        lowered = answer.lower()
        if lowered in ("y", "yes"):
            return True, "approved by the user"
        if lowered in ("a", "always"):
            self.always_allowed.add(tool.name)
            return True, "approved by the user (always, for this session)"
        if lowered in ("", "n", "no"):
            return False, "the user denied this action"
        return False, f"the user denied this action and said: {answer}"
