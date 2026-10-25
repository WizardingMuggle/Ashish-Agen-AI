"""
================================================================================
 MULTI-AGENT  —  a supervisor that delegates to specialists
================================================================================

One agent with 16 tools gets confused more easily than three agents with 3-6
tools each. The "supervisor" (or "orchestrator-worker") pattern:

                         SUPERVISOR  (tools: ask_researcher, ask_analyst, ask_operator)
                        /     |      \
               researcher   analyst   operator
               handbook,    inventory, tickets,
               employees,   calculator files
               dates

The trick is that a SPECIALIST IS JUST A TOOL for the supervisor ("agents as
tools"). The supervisor calls ask_analyst(task="..."), and the "tool" runs
a whole ReAct loop with its own system prompt, its own tools and its own fresh
message list, then returns its final report as the tool result.

Things to watch in the logs (each specialist's work is printed indented):
  * the supervisor must write a SELF-CONTAINED task: the specialist does not
    see the conversation, only the task string;
  * independent specialists are called in parallel, dependent ones in order;
  * cost: every specialist re-sends its own prompt + schemas. Multi-agent is
    clearer and more modular, not cheaper. Compare /stats with /mode react.
"""

from __future__ import annotations

from agent.config import AgentConfig
from agent.core.guardrails import ApprovalGate
from agent.core.react_agent import ReActAgent
from agent.core.trace import Trace
from agent.logger import log
from agent.prompts import SPECIALIST_ROLE, SPECIALISTS, agent_system_prompt, supervisor_system_prompt
from agent.tools import TOOL_GROUPS
from agent.tools.registry import Tool, ToolRegistry
from agent.usage import Budget


def build_supervisor(registry: ToolRegistry, cfg: AgentConfig, gate: ApprovalGate, trace: Trace,
                     budget: Budget, memory_facts: str | None) -> ReActAgent:
    team_registry = ToolRegistry()
    for name, expertise in SPECIALISTS.items():
        tools = [t for t in TOOL_GROUPS[name] if t not in cfg.disabled]
        role = SPECIALIST_ROLE.format(name=name, expertise=expertise)
        # Specialists don't get long-term memory: only the supervisor talks to the user.
        specialist = ReActAgent(name, agent_system_prompt(cfg, None, role=role), tools, registry, cfg, gate)
        team_registry.register(_delegation_tool(specialist, expertise, trace, budget))

    log.step(f"Team: supervisor + {len(SPECIALISTS)} specialists ({', '.join(SPECIALISTS)})")
    log.explain("Each specialist is wrapped as a tool. Calling the tool runs that specialist's own ReAct loop.")
    return ReActAgent("supervisor", supervisor_system_prompt(cfg, memory_facts),
                      team_registry.names(), team_registry, cfg, gate)


def _delegation_tool(specialist: ReActAgent, expertise: str, trace: Trace, budget: Budget) -> Tool:
    def delegate(task: str) -> str:
        with log.nested(specialist.name):
            result = specialist.run(task, [], trace, budget)
        note = "" if result.stop_reason == "answered" else f"\n(Note: stopped early: {result.stop_reason})"
        return f"Report from {specialist.name}:\n{result.answer}{note}"

    return Tool(
        name=f"ask_{specialist.name}",
        description=(f"Delegate a sub-task to the {specialist.name} specialist. {expertise} "
                     "The specialist does NOT see the conversation: write a complete, self-contained task."),
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string",
                                    "description": "The full task, including every name, ID and number needed."}},
            "required": ["task"],
            "additionalProperties": False,
        },
        func=delegate,
        side_effect=False,
        category="delegation",
    )
