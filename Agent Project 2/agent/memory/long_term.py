"""
================================================================================
 LONG-TERM MEMORY  —  facts that survive a restart
================================================================================

Short-term memory forgets everything when you quit. Long-term memory is a
small store on disk (.agent_state/long_term_memory.json) that the AGENT
ITSELF writes to, through a tool:

    You:   "Remember that I'm Priya, employee E102, and I like bullet points."
    Agent: calls remember(fact="User is Priya Natarajan, E102; prefers bullet points")

Next session, every saved fact is placed in the system prompt, so the agent
already knows who you are. With many facts you would not paste them all:
you would embed them and retrieve the relevant ones (RAG over memories, the
same thing you built in Project 1). The `recall` tool is a tiny keyword
version of that.

This is the same idea as "memory" in ChatGPT or Claude: the model decides
what is worth remembering, and the application decides where it is stored.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from agent.config import STATE_DIR

MEMORY_FILE = STATE_DIR / "long_term_memory.json"
MAX_FACTS_IN_PROMPT = 20


class LongTermMemory:
    def __init__(self) -> None:
        self.facts: list[dict] = []
        if MEMORY_FILE.exists():
            self.facts = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))

    def _save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(self.facts, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.facts = []
        self._save()

    def as_prompt_block(self) -> str:
        if not self.facts:
            return "(nothing saved yet)"
        return "\n".join(f"- {f['fact']}" for f in self.facts[-MAX_FACTS_IN_PROMPT:])

    # ------------------------------------------------------------------ #
    # The two tools the agent can call (registered in tools/__init__.py)
    # ------------------------------------------------------------------ #
    def remember(self, fact: str) -> str:
        """Save a fact about the user or their preferences to long-term memory, so it is
        known in future sessions. Only save stable, useful facts (who they are, their
        employee ID, how they like answers), never secrets.

        Args:
            fact: One short, self-contained sentence, e.g. "User is Priya Natarajan (E102)".
        """
        if any(f["fact"].lower() == fact.lower() for f in self.facts):
            return "Already remembered."
        self.facts.append({"fact": fact, "saved_at": datetime.now().isoformat(timespec="seconds")})
        self._save()
        return f"Saved to long-term memory ({len(self.facts)} facts stored)."

    def recall(self, query: str) -> str:
        """Search long-term memory for saved facts matching some keywords.

        Args:
            query: Keywords, e.g. "employee id".
        """
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        hits = [f["fact"] for f in self.facts
                if words & set(re.findall(r"[a-z0-9]+", f["fact"].lower()))]
        return "\n".join(hits) if hits else "No saved fact matches."
