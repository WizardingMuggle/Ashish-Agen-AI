"""
================================================================================
 CHAT HISTORY  —  short-term conversational memory
================================================================================

LLM APIs are STATELESS: the model remembers nothing between calls. A chatbot
"remembers" only because we send the previous turns again with every request.

History is used in two places:
  * Q2 query rewriting  — to resolve "it", "that", "and the drone?" etc.
  * Q9 prompt building  — so the answer can refer back to earlier answers.

We keep only the last `history_turns` turns: older turns cost tokens on every
request and rarely matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ChatHistory:
    turns: list[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question, answer))

    def recent(self, n: int) -> list[Turn]:
        return self.turns[-n:] if n > 0 else []

    def clear(self) -> None:
        self.turns.clear()

    def as_messages(self, n: int) -> list[dict[str, str]]:
        """Recent turns in the OpenAI chat format."""
        messages = []
        for t in self.recent(n):
            messages.append({"role": "user", "content": t.question})
            messages.append({"role": "assistant", "content": t.answer})
        return messages

    def __len__(self) -> int:
        return len(self.turns)
