"""
================================================================================
 LOGGER  —  the "teaching voice" of the project
================================================================================

Every module imports the single shared `log` object from here:

    from rag.logger import log
    log.stage("I3", "CHUNKING", "We split the document into small pieces ...")
    log.step("Created 42 chunks")
    log.explain("Smaller chunks = more precise retrieval, but less context.")

Each method has a minimum VERBOSITY level. The terminal only shows messages at
or below the current verbosity (change it live with `/set verbosity 1`).
The log FILE in logs/ always receives everything, so nothing is ever lost.

    level 0  answers + errors only
    level 1  stage headers + key results           (log.stage, log.step, log.success)
    level 2  tables + "WHY" explanations           (log.detail, log.explain, log.table)
    level 3  raw internals: prompts, vectors, math (log.debug, log.block(..., level=3))

We use the `rich` library for colours, tables and panels.
NOTE: text coming from the document can contain "[" and "]", which rich would
treat as style markup — so dynamic text is always passed through `esc()`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from rich.console import Console, RenderableType
from rich.markup import escape as esc
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

__all__ = ["log", "esc", "Table"]


class RagLogger:
    def __init__(self) -> None:
        self.verbosity = 3
        self.console = Console(highlight=False)
        self._file_console: Console | None = None
        self.log_file_path: Path | None = None

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
        if self.verbosity >= level:
            self.console.print(renderable)
        if self._file_console is not None:
            self._file_console.print(renderable)

    # ------------------------------------------------------------------ #
    # Big structural markers
    # ------------------------------------------------------------------ #
    def banner(self, title: str, subtitle: str = "") -> None:
        body = Text(title, style="bold white", justify="center")
        if subtitle:
            body.append("\n" + subtitle, style="italic grey70")
        self._emit(Panel(body, border_style="bright_blue", padding=(1, 4)), level=0)

    def stage(self, code: str, title: str, what: str = "") -> None:
        """A pipeline stage header, e.g.  [I3] CHUNKING."""
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
        """A teaching note: WHY is this happening / what does this number mean."""
        self._emit(f"    [bold magenta]WHY:[/] [magenta]{msg}[/]", level=2)

    def success(self, msg: str) -> None:
        self._emit(f"  [bold green]OK[/] {msg}", level=1)

    def warn(self, msg: str) -> None:
        self._emit(f"  [bold yellow]WARNING[/] [yellow]{msg}[/]", level=0)

    def error(self, msg: str) -> None:
        self._emit(f"  [bold red]ERROR[/] [red]{msg}[/]", level=0)

    def always(self, renderable: RenderableType) -> None:
        self._emit(renderable, level=0)

    def file_only(self, renderable: RenderableType) -> None:
        """For text already shown on screen another way (e.g. the streamed answer)."""
        if self._file_console is not None:
            self._file_console.print(renderable)

    # ------------------------------------------------------------------ #
    # Rich renderables
    # ------------------------------------------------------------------ #
    def table(self, table: Table, level: int = 2) -> None:
        self._emit(table, level=level)

    def block(self, title: str, text: str, level: int = 3, style: str = "grey50") -> None:
        """Show a raw block of text (a prompt, a chunk, an API payload...)."""
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
                  width: int = 40, max_value: float | None = None) -> None:
        """Tiny horizontal ASCII bar chart — great for visualising scores."""
        if not rows:
            return
        top = max_value if max_value is not None else max(v for _, v in rows) or 1.0
        lines = [f"    [bold]{esc(title)}[/]"]
        for label, value in rows:
            filled = int(round(width * max(value, 0) / top)) if top > 0 else 0
            bar = "#" * filled + "." * (width - filled)
            lines.append(f"    {esc(label):<10} [green]{bar}[/] {value:.4f}")
        self._emit("\n".join(lines), level=level)

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #
    @contextmanager
    def timer(self, label: str) -> Generator[dict[str, float], None, None]:
        """with log.timer("Embedding"): ...   -> prints how long it took."""
        result: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield result
        finally:
            result["seconds"] = time.perf_counter() - start
            self._emit(f"    [grey62]time: {label} took {result['seconds'] * 1000:.0f} ms[/]", level=1)


# The one shared logger every module uses.
log = RagLogger()
