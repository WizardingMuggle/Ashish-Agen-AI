"""
================================================================================
 TOOLS: company data  —  the agent's window into "internal systems"
================================================================================

In a real company these would call an HR API, an ERP, a SQL database...
Here they read two small JSON files in data/. The pattern is identical:
a tool is a thin, well-described wrapper around a system the model can't
reach by itself.

Design tips visible in this file:
  * Return structured data (dicts / lists): the registry turns it into JSON
    and models read JSON very reliably.
  * Accept fuzzy input ("priya", "e102", "battery") — models don't always
    use the exact ID — and say clearly when nothing matched.
  * Do the deterministic work in code: `list_low_stock` computes
    "below reorder point" so the model does not have to compare 8 numbers.
"""

from __future__ import annotations

import json
from functools import lru_cache

from agent.config import DATA_DIR


@lru_cache(maxsize=None)
def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def find_employee(query: str) -> list[dict] | str:
    """Find employees by ID (e.g. "E102") or by (part of) their name. Returns role, team,
    office, manager ID and the purchase date of their current laptop.

    Args:
        query: An employee ID or a full / partial name, e.g. "Priya".
    """
    q = query.strip().lower()
    hits = [e for e in _load("employees.json") if q == e["id"].lower() or q in e["name"].lower()]
    return hits or f"No employee matches '{query}'."


def list_team(team: str) -> list[dict] | str:
    """List every employee in a team (Operations, Autonomy or Drones).

    Args:
        team: The team name.
    """
    hits = [e for e in _load("employees.json") if e["team"].lower() == team.strip().lower()]
    teams = sorted({e["team"] for e in _load("employees.json")})
    return hits or f"No team called '{team}'. Teams: {teams}"


def check_inventory(query: str) -> list[dict] | str:
    """Look up stock for parts by SKU (e.g. "BAT-6S"), name or category (e.g. "battery").
    Returns stock, reorder point, unit cost (USD), suppliers, lead time and warehouse.

    Args:
        query: A SKU, part name or category.
    """
    q = query.strip().lower()
    hits = [p for p in _load("inventory.json")
            if q == p["sku"].lower() or q in p["name"].lower() or q in p["category"].lower()]
    return hits or f"No part matches '{query}'. Use list_low_stock or try a category."


def list_low_stock() -> list[dict]:
    """List every part whose stock is BELOW its reorder point, with how many units are
    missing to reach the target stock of 2 x reorder point."""
    low = []
    for p in _load("inventory.json"):
        if p["stock"] < p["reorder_point"]:
            low.append({**p, "units_to_target": 2 * p["reorder_point"] - p["stock"]})
    return low
