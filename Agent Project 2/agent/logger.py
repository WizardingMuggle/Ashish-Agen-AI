"""
================================================================================
 LOGGER  —  the "teaching voice" of the project
================================================================================

Same idea as in Project 1: every module imports the single shared `log`:

    from agent.logger import log
    log.stage("L1", "THINK", "The model decides what to do next ...")
    log.step("Model wants to call 2 tools")
    log.explain("Every step re-sends the whole conversation, so cost grows.")

Each method has a minimum VERBOSITY level. The terminal only shows messages at
or below the current verbosity (change it live with `/verbosity 1`).
The log FILE in logs/ always receives everything.

    level 0  answers, approvals, errors
    level 1  stage headers, tool calls, key results  (log.stage, log.step, log.success)
    level 2  tables + "WHY" explanations              (log.detail, log.explain, log.table)
    level 3  raw internals: messages, schemas, output (log.debug, log.block(..., level=3))

NEW in this project: `with log.nested("researcher"):` indents everything
inside it. In multi-agent mode a specialist's whole loop is printed indented
under the supervisor's step that called it, so you can see who did what.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from rich.console import Console, RenderableType
from rich.markup import escape as esc
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

__all__ = ["log", "esc", "Table"]

INDENT = 4


class AgentLogger:
    def __init__(self) -> None:
        self.verbosity = 3
        self.console = Console(highlight=False)
        self._file_console: Console | None = None
        self.log_file_path: Path | None = None
        self.depth = 0

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def enable_file_logging(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        self.log_file_path = folder / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"
        handle = open(self.log_file_path, "w", encoding="utf-8")
        self._file_console = Console(
            file=handle, width=130, color_system=None, highlight=False, soft_wrap=False
        )
        return self.log_file_path

    # ------------------------------------------------------------------ #
    # Core: print to terminal (if verbose enough) and always to the file
    # ------------------------------------------------------------------ #
    def _emit(self, renderable: RenderableType, level: int) -> None:
        if isinstance(renderable, str):
            renderable = Text.from_markup(renderable)
        if self.depth:
            renderable = Padding(renderable, (0, 0, 0, INDENT * self.depth))
        if self.verbosity >= level:
            self.console.print(renderable)
        if self._file_console is not None:
            self._file_console.print(renderable)

    @contextmanager
    def nested(self, name: str) -> Generator[None, None, None]:
        """Indent everything logged inside this block (a sub-agent's work)."""
        self._emit(f"[bold yellow]>> handing over to '{esc(name)}'[/]", level=1)
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            self._emit(f"[bold yellow]<< '{esc(name)}' finished, back to the caller[/]", level=1)

    # ------------------------------------------------------------------ #
    # Big structural markers
    # ------------------------------------------------------------------ #
    def banner(self, title: str, subtitle: str = "") -> None:
        body = Text(title, style="bold white", justify="center")
        if subtitle:
            body.append("\n" + subtitle, style="italic grey70")
        self._emit(Panel(body, border_style="bright_blue", padding=(1, 4)), level=0)

    def stage(self, code: str, title: str, what: str = "") -> None:
        """A stage header, e.g.  [L1] THINK."""
        self._emit(Text(""), level=1)
        self._emit(Rule(f"[bold bright_cyan]{esc(f'[{code}] {title}')}[/]", style="cyan", align="left"), level=1)
        if what:
            self._emit(Text("  " + what, style="italic grey62"), level=2)

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #
    def step(self, msg: str) -> None:
        self._emit(f"  [cyan]>[/] {msg}", level=1)

    def detail(self, msg: str) -> None:
        self._emit(f"    [grey70]-[/] {msg}", level=2)

    def debug(self, msg: str) -> None:
        self._emit(f"    [grey50]. {msg}[/]", level=3)

    def explain(self, msg: str) -> None:
        """A teaching note: WHY is this happening / what does this mean."""
        self._emit(f"    [bold magenta]WHY:[/] [magenta]{msg}[/]", level=2)

    def thought(self, agent: str, text: str) -> None:
        """The model's own reasoning sentence before it acts."""
        self._emit(f"  [bold yellow]THOUGHT ({esc(agent)}):[/] [yellow]{esc(text)}[/]", level=1)

    def success(self, msg: str) -> None:
        self._emit(f"  [bold green]OK[/] {msg}", level=1)

    def warn(self, msg: str) -> None:
        self._emit(f"  [bold yellow]WARNING[/] [yellow]{msg}[/]", level=0)

    def error(self, msg: str) -> None:
        self._emit(f"  [bold red]ERROR[/] [red]{msg}[/]", level=0)

    def always(self, renderable: RenderableType) -> None:
        self._emit(renderable, level=0)

    def file_only(self, renderable: RenderableType) -> None:
        """For text already shown on screen another way (e.g. what the user typed)."""
        if self._file_console is not None:
            self._file_console.print(renderable, markup=False)

    # ------------------------------------------------------------------ #
    # Rich renderables
    # ------------------------------------------------------------------ #
    def table(self, table: Table, level: int = 2) -> None:
        self._emit(table, level=level)

    def block(self, title: str, text: str, level: int = 3, style: str = "grey50") -> None:
        """Show a raw block of text (a prompt, a tool result, an API payload...)."""
        self._emit(
            Panel(Text(text), title=f"[bold]{esc(title)}[/]", title_align="left",
                  border_style=style, padding=(0, 1)),
            level=level,
        )

    def kv(self, title: str, data: dict[str, object], level: int = 2) -> None:
        """A two-column key/value table."""
        t = Table(title=title, title_justify="left", show_header=False, box=None, padding=(0, 2),
                  min_width=len(title) + 4)
        t.add_column(style="bold grey70")
        t.add_column()
        for k, v in data.items():
            t.add_row(str(k), esc(str(v)))
        self._emit(t, level=level)

    def bar_chart(self, title: str, rows: list[tuple[str, float]], level: int = 2,
                  width: int = 40, fmt: str = "{:,.0f}") -> None:
        """Tiny horizontal ASCII bar chart."""
        if not rows:
            return
        top = max(v for _, v in rows) or 1.0
        lines = [f"    [bold]{esc(title)}[/]"]
        for label, value in rows:
            filled = int(round(width * max(value, 0) / top))
            bar = "#" * filled + "." * (width - filled)
            lines.append(f"    {esc(label):<22} [green]{bar}[/] {fmt.format(value)}")
        self._emit("\n".join(lines), level=level)

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #
    @contextmanager
    def timer(self, label: str) -> Generator[dict[str, float], None, None]:
        """with log.timer("Tool call"): ...   -> prints how long it took."""
        result: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield result
        finally:
            result["seconds"] = time.perf_counter() - start
            self._emit(f"    [grey62]time: {esc(label)} took {result['seconds'] * 1000:.0f} ms[/]", level=2)


# The one shared logger every module uses.
log = AgentLogger()
