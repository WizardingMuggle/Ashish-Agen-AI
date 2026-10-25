"""
================================================================================
 REFLECTION  —  a second model checks the first one's work
================================================================================

Agents make mistakes that look confident: a number copied wrong, half of a
two-part question ignored, a ticket "created" that was never created.

Reflection (also called self-critique or "LLM-as-judge") adds a reviewer:

   [R1] CRITIQUE  a separate LLM call gets the question, the EVIDENCE (every
                  tool call and result from the trace) and the answer, and
                  returns {"verdict": "pass" | "revise", "issues": [...]}.
   [R2] REVISE    if the verdict is "revise", the issues are sent back to the
                  agent as a new user message. It keeps its working memory, may
                  call tools again, and writes an improved answer.

The critic sees the EVIDENCE, not just the answer. That is what lets it catch
invented facts: a critic that only sees the answer can only judge style.

Cost: at least one extra call per question, two or more if it revises.
Turn it on with:  /set use_reflection true
"""

from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.core.trace import Trace
from agent.llm import chat
from agent.logger import esc, log
from agent.prompts import CRITIC_PROMPT

CRITIC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "revise"]},
                "issues": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string", "description": "One sentence overall assessment."},
            },
            "required": ["verdict", "issues", "summary"],
            "additionalProperties": False,
        },
    },
}


def critique(question: str, answer: str, trace: Trace, cfg: AgentConfig, round_no: int) -> dict:
    log.stage(f"R{round_no}", "REFLECTION: CRITIQUE",
              "A separate reviewer call checks the answer against the evidence in the trace.")
    evidence = trace.evidence()
    messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}\n\nANSWER:\n{answer}"},
    ]
    log.block("Critic input", messages[1]["content"], level=3)
    choice = chat(messages, cfg, purpose=f"critic {round_no}", response_format=CRITIC_SCHEMA)
    review = json.loads(choice.message.content)
    trace.add("critic", "reflection", round=round_no, **review)

    colour = "green" if review["verdict"] == "pass" else "yellow"
    log.step(f"Verdict: [bold {colour}]{review['verdict'].upper()}[/]: {esc(review['summary'])}")
    for issue in review["issues"]:
        log.step(f"  [yellow]issue:[/] {esc(issue)}")
    return review
