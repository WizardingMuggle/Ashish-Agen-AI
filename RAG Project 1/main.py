"""
================================================================================
 RAG CHATBOT  —  terminal entry point
================================================================================

Run:
    python main.py                                   # default document + settings
    python main.py --file data/my_notes.txt          # ingest another workspace file
    python main.py --verbosity 1                     # quieter logs
    python main.py --mode bm25                       # start in keyword-only mode

Inside the chat, type a question, or a command (type /help to see them all).
The most useful command for learning is /set — change any setting live and
ask the same question again to see how the logs and the answer change.
"""

from __future__ import annotations

import argparse
import sys

from rich.text import Text

from rag.config import CHOICES, PROJECT_ROOT, REINDEX_KEYS, RAGConfig
from rag.diagram import FLOW_DIAGRAM
from rag.embeddings.openai_client import MissingAPIKeyError
from rag.logger import Table, esc, log
from rag.pipeline import RAGPipeline
from rag.utils.usage import format_cost, usage

HELP = [
    ("<your question>", "Ask anything about the document"),
    ("/help", "Show this help"),
    ("/config", "Show every setting and its current value"),
    ("/set <key> <value>", "Change a setting live, e.g. /set top_k 2  (re-indexes if needed)"),
    ("/verbosity <0-3>", "Shortcut for /set verbosity  (0 = answers only, 3 = everything)"),
    ("/search <query>", "Run retrieval only (no LLM call) and show the ranked chunks"),
    ("/chunks", "List all chunks"),
    ("/chunk <id>", "Show the full text of one chunk"),
    ("/term <word>", "Show what BM25 knows about a word (document frequency, IDF)"),
    ("/load <path>", "Ingest a different file from this workspace"),
    ("/reindex", "Rebuild the index from scratch"),
    ("/history", "Show the conversation memory"),
    ("/clear", "Forget the conversation (keeps the index)"),
    ("/stats", "Tokens and estimated cost of every API call this session"),
    ("/diagram", "Print the pipeline flow diagram"),
    ("/exit", "Quit"),
]


def print_help() -> None:
    t = Table(title="Commands", title_justify="left")
    t.add_column("command", style="bold cyan")
    t.add_column("what it does")
    for cmd, desc in HELP:
        t.add_row(esc(cmd), desc)
    log.always(t)


def print_config(cfg: RAGConfig) -> None:
    t = Table(title="Current settings (change with /set <key> <value>)", title_justify="left")
    t.add_column("key", style="bold")
    t.add_column("value", style="green")
    t.add_column("note", style="grey62")
    for key, value in cfg.as_dict().items():
        notes = []
        if key in REINDEX_KEYS:
            notes.append("re-index on change")
        if key in CHOICES:
            notes.append("one of " + ", ".join(map(str, CHOICES[key])))
        t.add_row(key, esc(str(value)), "; ".join(notes))
    log.always(t)


def print_stats() -> None:
    t = Table(title="API usage this session", title_justify="left")
    for col in ("#", "purpose", "model", "in tokens", "out tokens", "ms", "cost"):
        t.add_column(col)
    for i, c in enumerate(usage.calls, start=1):
        t.add_row(str(i), c.purpose, c.model, f"{c.input_tokens:,}", f"{c.output_tokens:,}",
                  f"{c.seconds * 1000:.0f}", format_cost(c.cost))
    log.always(t)
    log.always(f"  Total: {len(usage.calls)} calls, ~${usage.total_cost:.6f} (estimate)")


def handle_command(line: str, pipeline: RAGPipeline) -> bool:
    """Run a /command. Returns False when the user wants to quit."""
    cfg = pipeline.cfg
    cmd, _, arg = line.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()

    if cmd in ("/exit", "/quit"):
        return False
    elif cmd == "/help":
        print_help()
    elif cmd == "/config":
        print_config(cfg)
    elif cmd in ("/set", "/verbosity"):
        key, _, value = (f"verbosity {arg}" if cmd == "/verbosity" else arg).partition(" ")
        if not value:
            log.error("Usage: /set <key> <value>   (see /config for keys)")
            return True
        old = getattr(cfg, key, None)
        new = cfg.set_value(key, value)
        log.verbosity = cfg.verbosity
        log.always(f"  [green]{key}[/]: {esc(str(old))} -> [bold]{esc(str(new))}[/]")
        if key in REINDEX_KEYS:
            log.always("  This setting changes the index, so the document is re-indexed now...")
            pipeline.build_index()
        else:
            log.always("  Ask your question again to see the effect.")
    elif cmd == "/search":
        if not arg:
            log.error("Usage: /search <query>")
        else:
            pipeline.search_only(arg)
    elif cmd == "/chunks":
        t = Table(title=f"{len(pipeline.chunks)} chunks", title_justify="left")
        for col in ("id", "tokens", "section", "preview"):
            t.add_column(col, no_wrap=col != "preview")
        for c in pipeline.chunks:
            t.add_row(str(c.id), str(c.n_tokens), esc(c.section), esc(c.preview(80)))
        log.always(t)
    elif cmd == "/chunk":
        if not arg.isdigit() or int(arg) >= len(pipeline.chunks):
            log.error(f"Usage: /chunk <id>  with id between 0 and {len(pipeline.chunks) - 1}")
        else:
            c = pipeline.chunks[int(arg)]
            log.block(f"Chunk {c.id} | {c.n_tokens} tokens | section: {c.section} | "
                      f"chars {c.start_char}-{c.end_char}", c.text, level=0, style="cyan")
    elif cmd == "/term":
        if pipeline.bm25 is None or not arg:
            log.error("Usage: /term <word>")
        else:
            pipeline.bm25.describe_term(arg)
    elif cmd == "/load":
        if not arg:
            log.error("Usage: /load <path inside the workspace>")
        else:
            old = cfg.document_path
            cfg.document_path = arg
            try:
                pipeline.build_index()
                pipeline.history.clear()
            except Exception:
                cfg.document_path = old
                raise
    elif cmd == "/reindex":
        pipeline.build_index()
    elif cmd == "/history":
        if not pipeline.history.turns:
            log.always("  (no conversation yet)")
        for i, turn in enumerate(pipeline.history.turns, start=1):
            log.always(f"  [bold]{i}. You:[/] {esc(turn.question)}")
            log.always(f"     [green]Bot:[/] {esc(turn.answer[:300])}")
    elif cmd == "/clear":
        pipeline.history.clear()
        log.always("  Conversation memory cleared.")
    elif cmd == "/stats":
        print_stats()
    elif cmd == "/diagram":
        log.always(Text(FLOW_DIAGRAM, style="cyan"))
    else:
        log.error(f"Unknown command '{esc(cmd)}'. Type /help.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Educational RAG chatbot")
    parser.add_argument("--file", help="document to ingest (must be inside this project folder)")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], help="retrieval mode")
    args = parser.parse_args()

    cfg = RAGConfig()
    if args.file:
        cfg.document_path = args.file
    if args.verbosity is not None:
        cfg.verbosity = args.verbosity
    if args.mode:
        cfg.retrieval_mode = args.mode
    log.verbosity = cfg.verbosity
    if cfg.log_to_file:
        path = log.enable_file_logging(PROJECT_ROOT / "logs")
        log.always(f"[grey62]Full log of this session: {esc(str(path))}[/]")

    log.banner("RAG CHATBOT  -  learn Retrieval-Augmented Generation by watching it work",
               f"chat model: {cfg.chat_model}  |  embeddings: {cfg.embedding_model}  |  "
               f"verbosity: {cfg.verbosity}")
    if cfg.verbosity >= 2:
        log.always(Text(FLOW_DIAGRAM, style="cyan"))

    pipeline = RAGPipeline(cfg)
    try:
        pipeline.build_index()
    except MissingAPIKeyError as exc:
        log.error(str(exc))
        return 1
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        log.error(str(exc))
        return 1

    log.always("")
    log.always("[bold]Ready![/] Ask a question about the document, or type [cyan]/help[/]. "
               "Try: [italic]How many vacation days do I get?[/]")

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
                if not handle_command(line, pipeline):
                    log.always("Bye!")
                    break
            else:
                pipeline.ask(line)
        except KeyboardInterrupt:
            log.warn("Interrupted.")
        except (KeyError, ValueError, FileNotFoundError, PermissionError) as exc:
            log.error(esc(str(exc)))
        except Exception as exc:  # API errors etc. — keep the chat alive
            log.error(f"{type(exc).__name__}: {esc(str(exc))}")

    if usage.calls:
        print_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
