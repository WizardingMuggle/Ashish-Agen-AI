"""
================================================================================
 TRACE  —  a complete, replayable record of what the agent did
================================================================================

Debugging an agent is hard: the same question can take a different path every
time. Production teams therefore record a TRACE of every run: each LLM call,
each tool call with its arguments and result, each approval and each cost.
Tools like LangSmith, Langfuse or OpenAI's trace viewer are built on this idea.

Here every question produces a JSON file in traces/. Type `/trace` to see the
last one as a timeline, or open the file to see every detail.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Trace:
    question: str
    mode: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    events: list[dict] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def add(self, agent: str, kind: str, **data: object) -> None:
        """kind: llm | tool | approval | plan | reflection | final"""
        self.events.append({
            "t_ms": round((time.perf_counter() - self._t0) * 1000),
            "agent": agent,
            "kind": kind,
            **data,
        })

    def tool_events(self) -> list[dict]:
        return [e for e in self.events if e["kind"] == "tool"]

    def evidence(self, max_chars_per_result: int = 800) -> str:
        """Every tool call and its result, as text (used by the critic)."""
        lines = []
        for e in self.tool_events():
            lines.append(f"[{e['agent']}] {e['tool']}({e['arguments']}) ->\n{str(e['output'])[:max_chars_per_result]}")
        return "\n\n".join(lines) or "(the agent did not call any tools)"

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"trace_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = {"question": self.question, "mode": self.mode, "started_at": self.started_at,
                "events": self.events}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path
