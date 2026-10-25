"""
================================================================================
 AI AGENT  —  terminal entry point
================================================================================

Run:
    python main.py                       # default: single ReAct agent
    python main.py --mode plan           # plan-and-execute
    python main.py --mode multi          # supervisor + specialist agents
    python main.py --verbosity 1         # quieter logs
    python main.py --no-approval         # don't ask before tickets / file writes

Inside the chat, type a request, or a command (type /help to see them all).
The best commands for learning are /set (change a setting and ask again),
/run (call a tool yourself, exactly as the model would) and /trace.
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.text import Text

from agent.app import AgentApp
from agent.config import CHOICES, PROJECT_ROOT, AgentConfig
from agent.diagram import FLOW_DIAGRAM
from agent.llm import MissingAPIKeyError, get_client
from agent.logger import Table, esc, log
from agent.usage import format_cost, usage

HELP = [
    ("<your request>", "Ask or instruct the agent"),
    ("/help", "Show this help"),
    ("/config", "Show every setting and its current value"),
    ("/set <key> <value>", "Change a setting live, e.g. /set max_iterations 2"),
    ("/mode <react|plan|multi>", "Switch architecture (shortcut for /set agent_mode)"),
    ("/verbosity <0-3>", "0 = answers only, 3 = everything"),
    ("/tools", "List all tools, their category and whether they need approval"),
    ("/tool <name>", "Show the exact JSON schema the model receives for a tool"),
    ("/run <name> <json>", "Call a tool yourself, e.g. /run check_inventory {\"query\": \"battery\"}"),
    ("/disable <name>", "Hide a tool from the agent (see how it copes)"),
    ("/enable <name>", "Give a disabled tool back"),
    ("/trace", "Timeline of the last run: every LLM call, tool call and approval"),
    ("/memory", "Show long-term memory (facts saved with the remember tool)"),
    ("/forget", "Delete all long-term memory"),
    ("/history", "Show the short-term conversation memory"),
    ("/clear", "Forget the conversation (keeps long-term memory)"),
    ("/stats", "Tokens and estimated cost of every API call this session"),
    ("/diagram", "Print the flow diagram"),
    ("/exit", "Quit"),
]

EXAMPLES = [
    "What's 17.5% of 2,340 USD?",
    "Which parts are below their reorder point, and what would restocking them cost?",
    "Is Priya due for a laptop refresh? If yes, open a ticket for it.",
    "I'm flying to Berlin for 3 nights. What's my maximum hotel + meal budget?",
    "Write a restock report for the low-stock parts to restock_report.md",
    "Remember that I'm Alicia Gomez and I prefer bullet points.",
]


def print_help() -> None:
    t = Table(title="Commands", title_justify="left")
    t.add_column("command", style="bold cyan")
    t.add_column("what it does")
    for cmd, desc in HELP:
        t.add_row(esc(cmd), desc)
    log.always(t)
    log.always("  [bold]Try:[/] " + "\n       ".join(f"[italic]{esc(e)}[/]" for e in EXAMPLES))


def print_config(cfg: AgentConfig) -> None:
    t = Table(title="Current settings (change with /set <key> <value>)", title_justify="left")
    t.add_column("key", style="bold")
    t.add_column("value", style="green")
    t.add_column("allowed", style="grey62")
    for key, value in cfg.as_dict().items():
        allowed = ", ".join(map(str, CHOICES[key])) if key in CHOICES else ""
        t.add_row(key, esc(str(value)), allowed)
    log.always(t)


def print_tools(app: AgentApp) -> None:
    allowed = set(app.allowed_tools())
    t = Table(title=f"{len(app.registry.tools)} tools", title_justify="left")
    for col in ("name", "category", "approval", "visible", "description"):
        t.add_column(col, no_wrap=col != "description")
    for tool in app.registry.tools.values():
        t.add_row(tool.name, tool.category, "[yellow]yes[/]" if tool.side_effect else "no",
                  "yes" if tool.name in allowed else "[red]no[/]",
                  esc(tool.description[:90] + ("..." if len(tool.description) > 90 else "")))
    log.always(t)


def print_stats() -> None:
    t = Table(title="API usage this session", title_justify="left")
    for col in ("#", "purpose", "model", "in tokens", "out tokens", "ms", "cost"):
        t.add_column(col)
    for i, c in enumerate(usage.calls, start=1):
        t.add_row(str(i), esc(c.purpose), c.model, f"{c.input_tokens:,}", f"{c.output_tokens:,}",
                  f"{c.seconds * 1000:.0f}", format_cost(c.cost))
    log.always(t)
    log.always(f"  Total: {len(usage.calls)} calls, ~${usage.total_cost:.6f} (estimate)")


def set_tool_enabled(cfg: AgentConfig, app: AgentApp, name: str, enabled: bool) -> None:
    if app.registry.get(name) is None:
        log.error(f"No tool called '{esc(name)}'. Type /tools.")
        return
    disabled = cfg.disabled
    if enabled:
        disabled.discard(name)
    else:
        disabled.add(name)
    cfg.disabled_tools = ",".join(sorted(disabled))
    log.always(f"  {esc(name)} is now {'[green]enabled' if enabled else '[red]disabled'}[/]. "
               "Ask again to see how the agent copes.")


def handle_command(line: str, app: AgentApp) -> bool:
    """Run a /command. Returns False when the user wants to quit."""
    cfg = app.cfg
    cmd, _, arg = line.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()

    if cmd in ("/exit", "/quit"):
        return False
    elif cmd == "/help":
        print_help()
    elif cmd == "/config":
        print_config(cfg)
    elif cmd in ("/set", "/verbosity", "/mode"):
        if cmd == "/verbosity":
            arg = f"verbosity {arg}"
        elif cmd == "/mode":
            arg = f"agent_mode {arg}"
        key, _, value = arg.partition(" ")
        if not value:
            log.error("Usage: /set <key> <value>   (see /config for keys)")
            return True
        old = getattr(cfg, key, None)
        new = cfg.set_value(key, value)
        log.verbosity = cfg.verbosity
        log.always(f"  [green]{key}[/]: {esc(str(old))} -> [bold]{esc(str(new))}[/]. "
                   "Ask your question again to see the effect.")
    elif cmd == "/tools":
        print_tools(app)
    elif cmd == "/tool":
        tool = app.registry.get(arg)
        if tool is None:
            log.error("Usage: /tool <name>   (see /tools)")
        else:
            log.block(f"Schema of '{tool.name}' (exactly what the model sees)",
                      json.dumps(tool.schema(), indent=2), level=0, style="cyan")
    elif cmd == "/run":
        name, _, raw = arg.partition(" ")
        if app.registry.get(name) is None:
            log.error('Usage: /run <tool> <json args>, e.g. /run find_employee {"query": "Priya"}')
        else:
            result = app.registry.execute(name, raw.strip() or "{}", cfg.max_tool_output_chars)
            log.block(f"{name} -> {'ok' if result.ok else 'FAILED'} ({result.seconds * 1000:.0f} ms)",
                      result.output, level=0, style="green" if result.ok else "red")
            log.always("  [grey62]This exact string is what the model would receive as the tool result. "
                       "(/run skips the approval prompt: you are the human.)[/]")
    elif cmd == "/disable":
        set_tool_enabled(cfg, app, arg, enabled=False)
    elif cmd == "/enable":
        set_tool_enabled(cfg, app, arg, enabled=True)
    elif cmd == "/trace":
        app.show_trace()
    elif cmd == "/memory":
        log.always(f"  Long-term memory ({len(app.long_term.facts)} facts):\n{esc(app.long_term.as_prompt_block())}")
    elif cmd == "/forget":
        app.long_term.clear()
        log.always("  Long-term memory deleted.")
    elif cmd == "/history":
        if not app.history.turns:
            log.always("  (no conversation yet)")
        for i, turn in enumerate(app.history.turns, start=1):
            log.always(f"  [bold]{i}. You:[/] {esc(turn.question)}")
            log.always(f"     [green]Agent:[/] {esc(turn.answer[:300])}")
    elif cmd == "/clear":
        app.history.clear()
        log.always("  Conversation memory cleared.")
    elif cmd == "/stats":
        print_stats()
    elif cmd == "/diagram":
        log.always(Text(FLOW_DIAGRAM, style="cyan"))
    else:
        log.error(f"Unknown command '{esc(cmd)}'. Type /help.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Educational AI agent")
    parser.add_argument("--mode", choices=list(CHOICES["agent_mode"]), help="agent architecture")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--no-approval", action="store_true", help="don't ask before side-effect tools")
    args = parser.parse_args()

    cfg = AgentConfig()
    if args.mode:
        cfg.agent_mode = args.mode
    if args.verbosity is not None:
        cfg.verbosity = args.verbosity
    if args.no_approval:
        cfg.require_approval = False
    log.verbosity = cfg.verbosity
    if cfg.log_to_file:
        path = log.enable_file_logging(PROJECT_ROOT / "logs")
        log.always(f"[grey62]Full log of this session: {esc(str(path))}[/]")

    log.banner("AI AGENT  -  learn agentic AI by watching an agent think, act and observe",
               f"model: {cfg.chat_model}  |  mode: {cfg.agent_mode}  |  verbosity: {cfg.verbosity}")
    if cfg.verbosity >= 2:
        log.always(Text(FLOW_DIAGRAM, style="cyan"))

    try:
        get_client()
    except MissingAPIKeyError as exc:
        log.error(str(exc))
        return 1
    app = AgentApp(cfg)

    log.always("")
    log.always(f"[bold]Ready![/] {len(app.registry.tools)} tools loaded. Type a request, or [cyan]/help[/]. "
               f"Try: [italic]{esc(EXAMPLES[2])}[/]")

    while True:
        try:
            line = log.console.input("\n[bold blue]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            log.always("\nBye!")
            break
        if not line:
            continue
        log.file_only(f"\nYou: {line}")
        try:
            if line.startswith("/"):
                if not handle_command(line, app):
                    log.always("Bye!")
                    break
            else:
                app.ask(line)
        except KeyboardInterrupt:
            log.warn("Interrupted.")
        except (KeyError, ValueError) as exc:
            log.error(esc(str(exc)))
        except Exception as exc:  # API errors etc. — keep the chat alive
            log.error(f"{type(exc).__name__}: {esc(str(exc))}")

    if usage.calls:
        print_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
