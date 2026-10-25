"""
================================================================================
 THE AGENT APP  —  the conductor that runs one question end to end
================================================================================

This is the "table of contents" of the project, like rag/pipeline.py was.

 EVERY QUESTION
   S1  context        system prompt, long-term memory, allowed tools   (this file)
   then ONE architecture, chosen by agent_mode:
     react   L1..Ln   Think -> Act -> Observe loop      agent/core/react_agent.py
     plan    P1       plan (structured JSON)            agent/core/planner.py
             P2.k     execute step k with a ReAct agent
             P3       synthesize the answer
     multi   L1..Ln   supervisor loop; each ask_<specialist> call runs
                      that specialist's own loop (indented)  agent/core/supervisor.py
   R1..     reflection (optional): critique, maybe revise   agent/core/reflection.py
   F1       final answer, cost per step, trace saved        (this file)

 TOOLS        agent/tools/            (registry.py turns functions into schemas)
 GUARDRAILS   agent/core/guardrails.py (approval) + the loop (steps, budget)
 MEMORY       agent/memory/           (short-term turns, long-term facts on disk)
"""

from __future__ import annotations

import time
from typing import Callable

from agent.config import TRACES_DIR, AgentConfig
from agent.core.guardrails import ApprovalGate
from agent.core.planner import run_plan_and_execute
from agent.core.react_agent import AgentResult, ReActAgent
from agent.core.reflection import critique
from agent.core.supervisor import build_supervisor
from agent.core.trace import Trace
from agent.logger import Table, esc, log
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ConversationMemory
from agent.prompts import REVISE_FEEDBACK, agent_system_prompt
from agent.tools import MEMORY_TOOLS, build_registry
from agent.usage import Budget, format_cost


class AgentApp:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.history = ConversationMemory()
        self.long_term = LongTermMemory()
        self.registry = build_registry(self.long_term)
        self.gate = ApprovalGate(cfg)
        self.last_trace: Trace | None = None

    # ------------------------------------------------------------------ #
    # Which tools may the agent see right now?
    # ------------------------------------------------------------------ #
    def allowed_tools(self) -> list[str]:
        names = [n for n in self.registry.names() if n not in self.cfg.disabled]
        if not self.cfg.use_long_term_memory:
            names = [n for n in names if n not in MEMORY_TOOLS]
        return names

    # ------------------------------------------------------------------ #
    # One question
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> str:
        cfg = self.cfg
        start = time.perf_counter()
        trace = Trace(question, cfg.agent_mode)
        budget = Budget(cfg.max_cost_usd)

        # ---- S1: context ------------------------------------------------ #
        log.stage("S1", f"CONTEXT  (mode: {cfg.agent_mode})",
                  "Assemble what the agent starts with: instructions, memory, conversation and tools.")
        history = self.history.as_messages(cfg.history_turns)
        facts = self.long_term.as_prompt_block() if cfg.use_long_term_memory else None
        tools = self.allowed_tools()
        log.kv("Starting context", {
            "architecture": cfg.agent_mode,
            "earlier turns re-sent": f"{len(history) // 2} (history_turns={cfg.history_turns})",
            "long-term facts": len(self.long_term.facts) if facts is not None else "off",
            "tools available": f"{len(tools)} of {len(self.registry.names())}",
            "guardrails": f"max_iterations={cfg.max_iterations}, max_cost_usd=${cfg.max_cost_usd}, "
                          f"require_approval={cfg.require_approval}",
        }, level=1)
        if cfg.disabled:
            log.detail(f"Disabled tools (invisible to the model): {esc(', '.join(sorted(cfg.disabled)))}")

        # ---- the chosen architecture ------------------------------------ #
        result, revise = self._run_mode(question, history, facts, tools, trace, budget)

        # ---- R: reflection ---------------------------------------------- #
        if cfg.use_reflection:
            for round_no in range(1, cfg.max_reflection_rounds + 1):
                if budget.exceeded():
                    log.warn("Budget used up: skipping reflection.")
                    break
                review = critique(question, result.answer, trace, cfg, round_no)
                if review["verdict"] == "pass" or not review["issues"]:
                    break
                log.stage(f"R{round_no}", "REFLECTION: REVISE",
                          "The issues go back to the agent as a new message; it keeps its working memory.")
                issues = "\n".join(f"- {i}" for i in review["issues"])
                result = revise(REVISE_FEEDBACK.format(issues=issues))

        # ---- F1: answer + stats + trace --------------------------------- #
        self._finish(question, result, trace, budget, time.perf_counter() - start)
        return result.answer

    def _run_mode(self, question: str, history: list[dict], facts: str | None, tools: list[str],
                  trace: Trace, budget: Budget) -> tuple[AgentResult, Callable[[str], AgentResult]]:
        cfg = self.cfg
        if cfg.agent_mode == "plan":
            return run_plan_and_execute(question, history, agent_system_prompt(cfg, facts), tools,
                                        self.registry, cfg, self.gate, trace, budget)
        if cfg.agent_mode == "multi":
            agent = build_supervisor(self.registry, cfg, self.gate, trace, budget, facts)
            if cfg.use_long_term_memory:
                log.detail("multi mode: the supervisor sees saved facts but only specialists have tools, "
                           "so 'remember' is not available here.")
        else:
            agent = ReActAgent("agent", agent_system_prompt(cfg, facts), tools, self.registry, cfg, self.gate)

        result = agent.run(question, history, trace, budget)
        return result, lambda feedback: agent.continue_with(result, feedback, trace, budget)

    def _finish(self, question: str, result: AgentResult, trace: Trace, budget: Budget, seconds: float) -> None:
        log.stage("F1", "FINAL ANSWER + WHAT IT COST", "")
        log.always("")
        log.always("[bold green]Assistant:[/]")
        log.always(esc(result.answer))
        trace.add("app", "final", answer=result.answer, stop_reason=result.stop_reason,
                  cost_usd=round(budget.spent, 6))

        calls = budget.calls
        log.bar_chart("Prompt tokens per LLM call (watch it grow inside a loop)",
                      [(c.purpose, c.input_tokens) for c in calls], level=2)
        log.kv("Run summary", {
            "stop reason": result.stop_reason,
            "LLM calls": len(calls),
            "tool calls": len(trace.tool_events()),
            "tokens in / out": f"{sum(c.input_tokens for c in calls):,} / {sum(c.output_tokens for c in calls):,}",
            "estimated cost": format_cost(budget.spent),
            "total time": f"{seconds:.1f} s",
        }, level=1)
        if result.stop_reason != "answered":
            log.warn(f"The agent was stopped by a guardrail ({result.stop_reason}); the answer may be incomplete.")

        self.history.add(question, result.answer)
        self.last_trace = trace
        if self.cfg.save_traces:
            path = trace.save(TRACES_DIR)
            log.detail(f"Trace saved: {esc(str(path))}  (or type /trace)")

    # ------------------------------------------------------------------ #
    # /trace: the last run as a timeline
    # ------------------------------------------------------------------ #
    def show_trace(self) -> None:
        if self.last_trace is None:
            log.always("  (no trace yet: ask a question first)")
            return
        t = Table(title=f"Trace: {self.last_trace.question}", title_justify="left")
        for col in ("ms", "agent", "event", "details"):
            t.add_column(col, no_wrap=col != "details")
        for e in self.last_trace.events:
            if e["kind"] == "llm":
                calls = ", ".join(c["name"] for c in e.get("tool_calls", []))
                details = f"step {e.get('step', '-')}: " + (f"calls {calls}" if calls else "answers")
            elif e["kind"] == "tool":
                details = f"{e['tool']}({e['arguments']}) -> {'ok' if e['ok'] else 'FAILED'}"
            elif e["kind"] == "approval":
                details = f"{e['tool']}: {'allowed' if e['allowed'] else 'DENIED'} ({e['reason']})"
            elif e["kind"] == "plan":
                details = f"{len(e['plan']['steps'])} steps: {e['plan']['goal']}"
            elif e["kind"] == "reflection":
                details = f"{e['verdict']}: {e['summary']}"
            else:
                details = f"stop={e.get('stop_reason')} cost=${e.get('cost_usd')}"
            t.add_row(str(e["t_ms"]), e["agent"], e["kind"], esc(details[:160]))
        log.always(t)
