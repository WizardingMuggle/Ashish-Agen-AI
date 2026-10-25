"""
================================================================================
 SHORT-TERM MEMORY  —  the conversation so far
================================================================================

Inside ONE question, the agent's "working memory" is simply its message list:
system prompt, the question, and every tool call + tool result so far. That
list lives in core/react_agent.py and is thrown away when the answer is done.

ACROSS questions we keep only the final (question, answer) pairs. Why not keep
every tool call too? Because tool results are long: re-sending every result of
every earlier question would make each new question slower and more
expensive, and old details distract the model. The final answers usually
carry what matters. That is a TRADE-OFF: if the user asks "which tool did you
use for that?", the agent can't know. Try it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: list[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question, answer))

    def clear(self) -> None:
        self.turns.clear()

    def as_messages(self, last_n: int) -> list[dict]:
        """The last `last_n` turns as alternating user / assistant messages."""
        messages: list[dict] = []
        for turn in self.turns[-last_n:] if last_n > 0 else []:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages
