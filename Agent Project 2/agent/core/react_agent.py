"""
================================================================================
 THE ReAct LOOP  —  the heart of every AI agent  (read this file first)
================================================================================

RAG (Project 1) was a fixed PIPELINE: retrieve -> prompt -> answer, always the
same steps in the same order, decided by the programmer.

An AGENT is a LOOP in which the MODEL decides the next step:

        +--------------------------------------------------------------+
        |                                                              |
        v                                                              |
   [THINK]  send messages + tool schemas to the LLM                    |
        |                                                              |
        +-- reply has tool_calls? -- yes --> [ACT] run each tool       |
        |                                       |                      |
        |                                    [OBSERVE] append each     |
        |                                    result as a "tool"        |
        |                                    message  -----------------+
        |
        +-- no tool_calls --> that text IS the FINAL ANSWER. Stop.

This is called ReAct ("Reason + Act", Yao et al., 2022). The paper had the
model write "Thought: / Action: / Observation:" as plain text. Modern APIs do
the same thing with structured tool calls, which is what we use here.

The loop also enforces the guardrails:
  * max_iterations: at most N think steps, then the model must answer
  * max_cost_usd:   a hard budget for the whole question
  * approval:       tools with side effects ask a human first (guardrails.py)
  * allow-list:     the model can only run the tools THIS agent was given

The same class is reused everywhere: as the single agent ("react" mode), as
the step executor ("plan" mode), and as the supervisor AND every specialist
("multi" mode). An agent is just: a system prompt + a set of tools + this loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent.config import AgentConfig
from agent.core.guardrails import ApprovalGate
from agent.core.trace import Trace
from agent.llm import assistant_message_to_dict, chat
from agent.logger import Table, esc, log
from agent.tools.registry import ToolRegistry, ToolResult
from agent.usage import Budget, usage

FORCE_ANSWER = (
    "STOP: you have reached the {limit}. Do not call any more tools. Give your best final answer now, "
    "using only the information above, and say clearly what you could not finish."
)


@dataclass
class AgentResult:
    answer: str
    messages: list[dict]           # the full working memory of this run
    iterations: int
    stop_reason: str               # "answered" | "max_iterations" | "budget"
    tool_calls: int = 0
    prompt_tokens_per_step: list[int] = field(default_factory=list)


class ReActAgent:
    def __init__(self, name: str, system_prompt: str, tool_names: list[str],
                 registry: ToolRegistry, cfg: AgentConfig, gate: ApprovalGate) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tool_names = [n for n in tool_names if registry.get(n)]
        self.registry = registry
        self.cfg = cfg
        self.gate = gate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, task: str, history: list[dict], trace: Trace, budget: Budget) -> AgentResult:
        messages = [{"role": "system", "content": self.system_prompt}, *history,
                    {"role": "user", "content": task}]
        log.step(f"Agent [bold]{esc(self.name)}[/] starts with {len(self.tool_names)} tools: "
                 f"{esc(', '.join(self.tool_names)) or '(none)'}")
        log.block(f"Task given to '{self.name}'", task, level=2, style="blue")
        log.block(f"System prompt of '{self.name}'", self.system_prompt, level=3)
        return self._loop(messages, trace, budget, tool_calls=0, first_step=1)

    def continue_with(self, previous: AgentResult, feedback: str, trace: Trace, budget: Budget) -> AgentResult:
        """Keep the same working memory and add a new user message (used for revisions)."""
        messages = [*previous.messages, {"role": "user", "content": feedback}]
        return self._loop(messages, trace, budget, tool_calls=previous.tool_calls,
                          first_step=previous.iterations + 1)

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def _loop(self, messages: list[dict], trace: Trace, budget: Budget, tool_calls: int,
              first_step: int) -> AgentResult:
        cfg = self.cfg
        schemas = self.registry.schemas(self.tool_names)
        tokens_per_step: list[int] = []
        stop_reason = "max_iterations"

        last_step = first_step + cfg.max_iterations - 1
        for step in range(first_step, last_step + 1):
            # ---- guardrail: budget ------------------------------------- #
            if budget.exceeded():
                log.warn(f"Budget of ${cfg.max_cost_usd} for this question is used up "
                         f"(${budget.spent:.4f} spent). Forcing a final answer.")
                stop_reason = "budget"
                break

            # ---- THINK ------------------------------------------------- #
            log.stage(f"L{step}", f"{self.name.upper()} STEP {step}/{last_step}: THINK",
                      "Send the whole conversation + tool schemas; the model decides: call tools, or answer.")
            self._show_context(messages, schemas, step)
            choice = chat(messages, cfg, purpose=f"{self.name} step {step}", tools=schemas or None)
            msg = choice.message
            tokens_per_step.append(usage.calls[-1].input_tokens)
            trace.add(self.name, "llm", step=step, prompt_tokens=usage.calls[-1].input_tokens,
                      content=msg.content,
                      tool_calls=[{"name": t.function.name, "arguments": t.function.arguments}
                                  for t in msg.tool_calls or []])

            if not msg.tool_calls:
                # ---- no tool call = FINAL ANSWER ------------------------ #
                answer = msg.content or ""
                messages.append({"role": "assistant", "content": answer})
                if choice.finish_reason == "length":
                    log.warn("The reply was cut off by max_output_tokens.")
                log.success(f"No tool call in the reply, so this is [bold]{esc(self.name)}[/]'s FINAL ANSWER "
                            f"(at step {step}, {tool_calls} tool call(s) so far).")
                log.explain("The model, not our code, decided it was done. That is the key difference from a fixed pipeline.")
                return AgentResult(answer, messages, step, "answered", tool_calls, tokens_per_step)

            if msg.content:
                log.thought(self.name, msg.content)
            else:
                log.detail("(the model called tools without writing a thought)")
            messages.append(assistant_message_to_dict(msg))

            # ---- ACT + OBSERVE ----------------------------------------- #
            n = len(msg.tool_calls)
            log.stage(f"L{step}", f"{self.name.upper()} STEP {step}: ACT -> OBSERVE",
                      "Our code runs the requested tools and appends each result as a 'tool' message.")
            if n > 1:
                log.explain(f"The model asked for {n} tools in ONE reply (parallel tool calls): "
                            "one round trip instead of several.")
            for call in msg.tool_calls:
                result = self._run_tool(call.id, call.function.name, call.function.arguments, trace)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.output})
                tool_calls += 1
        else:
            log.warn(f"'{self.name}' used all {cfg.max_iterations} steps without finishing. Forcing a final answer.")

        # ---- guardrail hit: one last call with tools switched off ---------- #
        limit = "cost budget" if stop_reason == "budget" else "maximum number of steps"
        log.stage("L!", f"{self.name.upper()}: FORCED FINAL ANSWER",
                  "tool_choice='none' makes the API refuse any more tool calls, so the model must answer.")
        messages.append({"role": "user", "content": FORCE_ANSWER.format(limit=limit)})
        choice = chat(messages, cfg, purpose=f"{self.name} forced answer",
                      tools=schemas or None, tool_choice="none" if schemas else None)
        answer = choice.message.content or ""
        messages.append({"role": "assistant", "content": answer})
        trace.add(self.name, "llm", step="forced", content=answer)
        return AgentResult(answer, messages, last_step, stop_reason, tool_calls, tokens_per_step)

    # ------------------------------------------------------------------ #
    # One tool call: allow-list -> approval -> execute -> observe
    # ------------------------------------------------------------------ #
    def _run_tool(self, call_id: str, name: str, raw_arguments: str, trace: Trace) -> ToolResult:
        log.step(f"[bold]ACT[/] {esc(self.name)} -> [bold green]{esc(name)}[/]({esc(_short_args(raw_arguments))})")
        log.debug(f"tool_call_id={call_id}  raw arguments as written by the model: {esc(raw_arguments)}")

        tool = self.registry.get(name)
        if tool is None or name not in self.tool_names:
            result = ToolResult(False, f"ERROR: you do not have a tool called '{name}'. "
                                       f"Your tools are: {self.tool_names}", 0.0)
        elif tool.side_effect and self.registry.parse_arguments(name, raw_arguments)[1]:
            # Malformed call: report the error without bothering the human.
            result = self.registry.execute(name, raw_arguments, self.cfg.max_tool_output_chars)
        else:
            allowed, reason = self.gate.check(tool, raw_arguments)
            if tool.side_effect:
                trace.add(self.name, "approval", tool=name, allowed=allowed, reason=reason)
                log.detail(f"approval: {'ALLOWED' if allowed else 'DENIED'} ({esc(reason)})")
            if allowed:
                result = self.registry.execute(name, raw_arguments, self.cfg.max_tool_output_chars)
            else:
                result = ToolResult(False, f"DENIED: {reason}. The action was NOT performed. "
                                           "Do not repeat the same call. If the user said what to change, "
                                           "you may try once more with that change; otherwise tell the user "
                                           "what you would have done.", 0.0)

        colour = "green" if result.ok else "red"
        preview = " ".join(result.output.split())[:160]
        log.step(f"[bold]OBSERVE[/] [{colour}]{'ok' if result.ok else 'failed'}[/] "
                 f"({result.seconds * 1000:.0f} ms, {len(result.output):,} chars): {esc(preview)}")
        log.block(f"Full result of {name}", result.output, level=3, style=colour)
        if result.truncated:
            log.detail(f"Output was cut to max_tool_output_chars={self.cfg.max_tool_output_chars}.")
        if not result.ok:
            log.explain("Failures are not exceptions in an agent: the message goes back to the model as the "
                        "tool result, and the model can correct itself in the next step.")
        trace.add(self.name, "tool", tool=name, arguments=raw_arguments, ok=result.ok,
                  output=result.output, ms=round(result.seconds * 1000))
        return result

    # ------------------------------------------------------------------ #
    # Teaching output: what exactly is being sent this step?
    # ------------------------------------------------------------------ #
    def _show_context(self, messages: list[dict], schemas: list[dict], step: int) -> None:
        counts: dict[str, int] = {}
        for m in messages:
            counts[m["role"]] = counts.get(m["role"], 0) + 1
        log.detail(f"Sending {len(messages)} messages ({', '.join(f'{v} {k}' for k, v in counts.items())}) "
                   f"+ {len(schemas)} tool schemas")
        if step == 1:
            log.explain("Tool schemas are sent with EVERY call: they are how the model knows what it can do.")
            log.block("Tool schemas sent to the model (JSON)", json.dumps(schemas, indent=1), level=3)
        else:
            log.explain("The prompt grows every step: all earlier tool calls and results are re-sent. "
                        "The model has no memory between calls except these messages.")
        if log.verbosity >= 3:
            t = Table(title="Messages in this request", title_justify="left")
            for col in ("#", "role", "content"):
                t.add_column(col, no_wrap=col != "content")
            for i, m in enumerate(messages):
                if m.get("tool_calls"):
                    content = "TOOL CALLS: " + ", ".join(
                        f"{c['function']['name']}({_short_args(c['function']['arguments'])})" for c in m["tool_calls"])
                    if m.get("content"):
                        content = f"{m['content']} | {content}"
                else:
                    content = str(m.get("content") or "")
                content = " ".join(content.split())
                t.add_row(str(i), m["role"], esc(content[:180] + ("..." if len(content) > 180 else "")))
            log.table(t, level=3)


def _short_args(raw: str, limit: int = 90) -> str:
    text = " ".join(str(raw).split())
    return text if len(text) <= limit else text[:limit] + "..."
