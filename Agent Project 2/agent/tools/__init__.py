"""All tools, registered in one place. `build_registry()` is called once at start-up."""

from __future__ import annotations

from agent.memory.long_term import LongTermMemory
from agent.tools import actions, calculator, company_data, knowledge
from agent.tools.registry import Tool, ToolRegistry, ToolResult

__all__ = ["build_registry", "Tool", "ToolRegistry", "ToolResult", "TOOL_GROUPS"]


def build_registry(long_term: LongTermMemory) -> ToolRegistry:
    registry = ToolRegistry()
    # --- read-only: safe to run without asking -------------------------- #
    registry.add(calculator.calculator, category="maths & time")
    registry.add(calculator.get_current_datetime, category="maths & time")
    registry.add(calculator.months_between, category="maths & time")
    registry.add(knowledge.search_handbook, category="knowledge")
    registry.add(knowledge.list_handbook_sections, category="knowledge")
    registry.add(company_data.find_employee, category="company data")
    registry.add(company_data.list_team, category="company data")
    registry.add(company_data.check_inventory, category="company data")
    registry.add(company_data.list_low_stock, category="company data")
    registry.add(actions.list_tickets, category="actions")
    registry.add(actions.list_workspace_files, category="actions")
    registry.add(actions.read_workspace_file, category="actions")
    # --- side effects: a human must approve (if require_approval) ------- #
    registry.add(actions.create_ticket, side_effect=True, category="actions")
    registry.add(actions.write_workspace_file, side_effect=True, category="actions")
    # --- memory: bound methods work just like functions ----------------- #
    registry.add(long_term.remember, category="memory")
    registry.add(long_term.recall, category="memory")
    return registry


# Which tools each specialist gets in multi-agent mode (core/supervisor.py).
# Giving an agent FEWER tools makes it pick the right one more reliably.
TOOL_GROUPS: dict[str, list[str]] = {
    "researcher": ["search_handbook", "list_handbook_sections", "find_employee", "list_team",
                   "get_current_datetime", "months_between"],
    "analyst": ["check_inventory", "list_low_stock", "calculator"],
    "operator": ["list_tickets", "create_ticket", "list_workspace_files",
                 "read_workspace_file", "write_workspace_file"],
}
MEMORY_TOOLS = ["remember", "recall"]
