"""
================================================================================
 TOOLS: actions  —  tools that CHANGE things (tickets and files)
================================================================================

Reading data is safe. Writing data is where agents become risky: a model that
misunderstood the task can open 20 duplicate tickets or overwrite a file.

So these tools are registered with `side_effect=True`, and the agent loop
asks a human before running them (see core/guardrails.py). That pattern is
called HUMAN-IN-THE-LOOP, and every production agent that acts on the
world uses some version of it.

A second guardrail lives in the tools themselves: the file tools can only
touch files inside workspace/. A path like "../.env" is rejected, no matter
what the model asks for. NEVER rely on the prompt alone for security: the
model can be confused or manipulated (prompt injection), the code cannot.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from agent.config import STATE_DIR, WORKSPACE_DIR

TICKETS_FILE = STATE_DIR / "tickets.json"
MAX_FILE_CHARS = 20_000


# ---------------------------------------------------------------------- #
# Support tickets
# ---------------------------------------------------------------------- #
def _load_tickets() -> list[dict]:
    if not TICKETS_FILE.exists():
        return []
    return json.loads(TICKETS_FILE.read_text(encoding="utf-8"))


def create_ticket(title: str, description: str, priority: Literal["P1", "P2", "P3"],
                  requester_id: str) -> str:
    """Open a new support ticket. Check list_tickets first so you don't create a duplicate.

    Args:
        title: Short title, e.g. "Laptop refresh".
        description: What is needed and why, including relevant facts (dates, IDs, amounts).
        priority: P1 = critical, P2 = blocks a person/team, P3 = normal request.
        requester_id: Employee ID of the person the ticket is for, e.g. "E102".
    """
    tickets = _load_tickets()
    ticket = {
        "id": f"T-{1001 + len(tickets)}",
        "title": title,
        "description": description,
        "priority": priority,
        "requester_id": requester_id,
        "status": "open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    tickets.append(ticket)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    return f"Created ticket {ticket['id']} ({priority}): {title}"


def list_tickets(status: Literal["open", "closed", "all"] = "open") -> list[dict] | str:
    """List support tickets.

    Args:
        status: Which tickets to show.
    """
    tickets = [t for t in _load_tickets() if status == "all" or t["status"] == status]
    return tickets or f"There are no {status} tickets."


# ---------------------------------------------------------------------- #
# Workspace files
# ---------------------------------------------------------------------- #
def _safe_path(relative: str) -> Path:
    """Resolve `relative` inside workspace/ and refuse anything outside it."""
    path = (WORKSPACE_DIR / relative).resolve()
    if not path.is_relative_to(WORKSPACE_DIR.resolve()):
        raise PermissionError(f"'{relative}' is outside the workspace folder. Use a plain file name.")
    return path


def list_workspace_files() -> str:
    """List the files in the agent's workspace folder."""
    files = [p for p in WORKSPACE_DIR.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if not files:
        return "The workspace is empty."
    return "\n".join(f"{p.relative_to(WORKSPACE_DIR).as_posix()} ({p.stat().st_size} bytes)" for p in files)


def read_workspace_file(path: str) -> str:
    """Read a text file from the workspace folder.

    Args:
        path: File name relative to the workspace, e.g. "restock_report.md".
    """
    return _safe_path(path).read_text(encoding="utf-8")[:MAX_FILE_CHARS]


def write_workspace_file(path: str, content: str) -> str:
    """Create or overwrite a text file (e.g. a Markdown report) in the workspace folder.

    Args:
        path: File name relative to the workspace, e.g. "restock_report.md".
        content: The full text to write.
    """
    target = _safe_path(path)
    if len(content) > MAX_FILE_CHARS:
        raise ValueError(f"content is too long ({len(content)} characters, max {MAX_FILE_CHARS})")
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"{'Overwrote' if existed else 'Created'} workspace/{path} ({len(content)} characters)"
