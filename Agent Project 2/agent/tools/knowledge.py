"""
================================================================================
 TOOLS: handbook search  —  RAG as a tool ("agentic RAG")
================================================================================

In Project 1, retrieval ran on EVERY question, always with the user's words
as the query. Here retrieval is just one tool among many, and the MODEL decides:
  * WHETHER to search at all (a maths question needs no handbook),
  * WHAT to search for (it writes its own query, often better than the user's),
  * WHETHER to search AGAIN with a different query if the first result was poor.
That is what people mean by "agentic RAG".

To keep this project focused on agents, the search is a simple keyword scorer
over the handbook's sections. Swap in the hybrid retriever from Project 1 and
nothing else in the agent has to change: the tool's schema stays the same.
"""

from __future__ import annotations

import re
from functools import lru_cache

from agent.config import DATA_DIR

HANDBOOK = DATA_DIR / "ops_handbook.md"
_STOPWORDS = set("a an and are as at be by can do does for from how i if in is it my of on or "
                 "the to up what when which who why with you your".split())


def _words(text: str) -> list[str]:
    return [w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


@lru_cache(maxsize=1)
def _sections() -> list[tuple[str, str]]:
    """Split the handbook into (section path, text) pairs on ## / ### headings."""
    sections: list[tuple[str, str]] = []
    h2 = h3 = ""
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((f"{h2} > {h3}" if h3 else h2 or "Introduction", body))
        lines.clear()

    for line in HANDBOOK.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            flush()
            h3 = line[4:].strip()
        elif line.startswith("## "):
            flush()
            h2, h3 = line[3:].strip(), ""
        elif not line.startswith("# "):
            lines.append(line)
    flush()
    return sections


def search_handbook(query: str, top_k: int = 3) -> str:
    """Search the Nimbus Robotics operations handbook (company policies: travel, hotel caps,
    meal per diem, approvals, laptop refresh, equipment, inventory reorder rules, ticket
    priorities and SLAs, offices and time zones). Returns the best matching sections.

    Args:
        query: Keywords describing what you are looking for, e.g. "hotel cap Berlin".
        top_k: How many sections to return (1-5).
    """
    top_k = max(1, min(top_k, 5))
    query_words = set(_words(query))
    scored = []
    for title, body in _sections():
        title_words, body_words = _words(title), _words(body)
        score = sum(2 * title_words.count(w) + body_words.count(w) for w in query_words)
        if score:
            scored.append((score, title, body))
    if not scored:
        return f"No handbook section matches '{query}'. Try different keywords."
    scored.sort(key=lambda s: -s[0])
    return "\n\n".join(f"[{title}] (score {score})\n{body}" for score, title, body in scored[:top_k])


def list_handbook_sections() -> str:
    """List the titles of all handbook sections, to see what policies exist."""
    return "\n".join(title for title, _ in _sections())
