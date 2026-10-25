"""
================================================================================
 PLAN-AND-EXECUTE  —  think first, then act
================================================================================

A ReAct agent decides ONE step at a time. That is flexible, but on longer tasks
it can wander, repeat itself or forget part of the request.

Plan-and-execute splits the work into three phases:

   [P1] PLAN        one LLM call writes a numbered list of steps.
                    We force the reply into a JSON schema ("structured
                    outputs"), so our code can read it reliably: no regex
                    parsing of free text.
   [P2] EXECUTE     for each step, a fresh ReAct agent runs with the goal,
                    the results so far, and only that step as its task.
   [P3] SYNTHESIZE  one LLM call writes the final answer from the step results.

Trade-offs you can observe in the logs (compare with /mode react):
  + the plan is visible up front: easy to review, easy to explain
  + each executor sees a short, focused context
  - more LLM calls, so usually slower and more expensive
  - the plan is fixed: if step 2 reveals something unexpected, the plan does
    not change. ("Re-planning" after each step is a challenge exercise.)
"""

from __future__ import annotations

import json
from typing import Callable

from agent.config import AgentConfig
from agent.core.guardrails import ApprovalGate
from agent.core.react_agent import AgentResult, ReActAgent
from agent.core.trace import Trace
from agent.llm import chat
from agent.logger import Table, esc, log
from agent.prompts import PLANNER_PROMPT, STEP_TASK, SYNTHESIZER_PROMPT
from agent.tools.registry import ToolRegistry
from agent.usage import Budget

# JSON Schema for the plan. strict=True: the API guarantees the reply matches it.
PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The user's goal in one sentence."},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"},
                                      "description": "Tools this step will probably use."},
                        },
                        "required": ["description", "tools"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["goal", "steps"],
            "additionalProperties": False,
        },
    },
}
MAX_STEPS = 6


def make_plan(question: str, history: list[dict], tool_names: list[str], registry: ToolRegistry,
              cfg: AgentConfig, trace: Trace) -> dict:
    log.stage("P1", "PLAN", "One LLM call turns the request into an ordered list of steps (JSON).")
    catalog = "\n".join(f"- {n}: {registry.get(n).description.split('. ')[0]}" for n in tool_names)
    messages = [{"role": "system", "content": PLANNER_PROMPT.format(tools=catalog)},
                *history, {"role": "user", "content": question}]
    log.block("Planner prompt", messages[0]["content"], level=3)
    log.explain("response_format=json_schema (strict) forces valid JSON in exactly this shape, so code can loop over the steps.")
    choice = chat(messages, cfg, purpose="planner", response_format=PLAN_SCHEMA)
    plan = json.loads(choice.message.content)
    plan["steps"] = plan["steps"][:MAX_STEPS]
    trace.add("planner", "plan", plan=plan)

    t = Table(title=f"Plan for: {plan['goal']}", title_justify="left")
    for col in ("#", "step", "expected tools"):
        t.add_column(col)
    for i, s in enumerate(plan["steps"], start=1):
        t.add_row(str(i), esc(s["description"]), esc(", ".join(s["tools"])))
    log.table(t, level=1)
    log.block("Raw plan JSON", json.dumps(plan, indent=2), level=3)
    return plan


def run_plan_and_execute(question: str, history: list[dict], system_prompt: str, tool_names: list[str],
                         registry: ToolRegistry, cfg: AgentConfig, gate: ApprovalGate,
                         trace: Trace, budget: Budget) -> tuple[AgentResult, Callable[[str], AgentResult]]:
    plan = make_plan(question, history, tool_names, registry, cfg, trace)
    if not plan["steps"]:
        plan["steps"] = [{"description": question, "tools": []}]

    results: list[str] = []
    total_tool_calls = 0
    for i, step in enumerate(plan["steps"], start=1):
        log.stage(f"P2.{i}", f"EXECUTE STEP {i}/{len(plan['steps'])}: {step['description']}",
                  "A fresh ReAct agent does just this step. It sees the goal and earlier results, not the full chat.")
        if budget.exceeded():
            log.warn("Budget used up: skipping the remaining steps.")
            results.append(f"Step {i} ({step['description']}): NOT DONE, budget exhausted.")
            continue
        done = "\n".join(results) or "(none yet)"
        task = STEP_TASK.format(goal=plan["goal"], done=done, number=i, total=len(plan["steps"]),
                                step=step["description"])
        executor = ReActAgent(f"executor-{i}", system_prompt, tool_names, registry, cfg, gate)
        with log.nested(f"executor-{i}"):
            result = executor.run(task, [], trace, budget)
        total_tool_calls += result.tool_calls
        results.append(f"Step {i} ({step['description']}): {result.answer}")
        log.success(f"Step {i} result: {esc(' '.join(result.answer.split())[:200])}")

    log.stage("P3", "SYNTHESIZE", "One last LLM call (no tools) writes the answer from the step results.")
    messages = [{"role": "system", "content": SYNTHESIZER_PROMPT},
                {"role": "user", "content": f"User request: {question}\n\nPlan results:\n" + "\n\n".join(results)}]
    log.block("Synthesizer input", messages[1]["content"], level=3)

    def synthesize(msgs: list[dict]) -> AgentResult:
        choice = chat(msgs, cfg, purpose="synthesizer")
        answer = choice.message.content or ""
        msgs = [*msgs, {"role": "assistant", "content": answer}]
        trace.add("synthesizer", "llm", content=answer)
        return AgentResult(answer, msgs, len(plan["steps"]), "answered", total_tool_calls)

    first = synthesize(messages)

    def revise(feedback: str) -> AgentResult:
        # The synthesizer has no tools, so a revision can only rewrite the answer
        # from the same step results. (Re-planning is a challenge exercise.)
        log.detail("Plan mode: the revision rewrites the answer from the same step results (no new tool calls).")
        return synthesize([*first.messages, {"role": "user", "content": feedback}])

    return first, revise
