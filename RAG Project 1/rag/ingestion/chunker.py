"""
================================================================================
 INDEXING STEP I3  —  CHUNKING
================================================================================

We cut the cleaned document into small, overlapping pieces called CHUNKS.
A chunk is the unit we search for, and the unit we paste into the prompt.

    Document (5,000 tokens)
    +-----------+-----------+-----------+-----------+
    |  chunk 0  |  chunk 1  |  chunk 2  |  chunk 3  |  ...
    +-----------+-----------+-----------+-----------+
              <-->        <-->        <-->
            overlap     overlap     overlap

The two big knobs:
  * chunk_size     too small -> a chunk lacks context ("it lasts 38 minutes" — what does?)
                   too big   -> one chunk mixes many topics, its embedding becomes
                                a blurry "average", and prompts get expensive.
  * chunk_overlap  repeats the end of chunk N at the start of chunk N+1, so a fact
                   that sits exactly on a boundary still appears whole somewhere.

Two strategies are implemented:

  "fixed"      Slide a window of `chunk_size` tokens, stepping by
               (chunk_size - chunk_overlap). Simple, but cuts mid-sentence.

  "recursive"  Split on the most "natural" boundary that keeps pieces small
               enough: blank line (paragraph) -> newline -> sentence -> word.
               Pieces are cut no bigger than chunk_overlap, so the overlap can be
               made of whole sentences. Then neighbouring pieces are glued back
               together up to chunk_size. Chunks therefore end on sentence or
               paragraph boundaries instead of mid-word.

Every chunk also remembers which markdown SECTION it came from
("Leave Policy > Sick Leave"). With `prepend_section_title` that path is added
to the text we embed, so a chunk that only says "12 days per year" still
carries the meaning "sick leave".
"""

from __future__ import annotations

import re
import statistics
from bisect import bisect_right
from dataclasses import dataclass

from rag.config import RAGConfig
from rag.ingestion.loader import Document
from rag.logger import Table, esc, log
from rag.utils.tokens import count_tokens, get_encoding

# Natural boundaries, from "strongest" to "weakest".
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: int
    text: str            # the exact text from the document
    start_char: int      # where it starts in the cleaned document
    end_char: int        # where it ends (exclusive)
    n_tokens: int
    section: str         # markdown heading path, e.g. "Leave Policy > Sick Leave"
    source: str          # file name

    def embed_text(self, prepend_section: bool) -> str:
        """The text that is actually sent to the embedding model."""
        if prepend_section and self.section:
            return f"Section: {self.section}\n{self.text}"
        return self.text

    def preview(self, n: int = 90) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= n else flat[: n - 3] + "..."


# ---------------------------------------------------------------------- #
# Section tracking (markdown headings)
# ---------------------------------------------------------------------- #
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _section_index(text: str) -> tuple[list[int], list[str]]:
    """Return (positions, paths): the heading path that is active from each position on."""
    positions, paths, stack = [], [], []  # stack of (level, title)
    for m in HEADING_RE.finditer(text):
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        positions.append(m.start())
        # Skip the level-1 document title, it is the same for every chunk.
        paths.append(" > ".join(t for lvl, t in stack if lvl > 1) or title)
    return positions, paths


def _section_at(pos: int, positions: list[int], paths: list[str]) -> str:
    i = bisect_right(positions, pos) - 1
    return paths[i] if i >= 0 else ""


# ---------------------------------------------------------------------- #
# Strategy 1: fixed-size token windows
# ---------------------------------------------------------------------- #
def _fixed_spans(text: str, size: int, overlap: int, model: str) -> list[tuple[int, int]]:
    enc = get_encoding(model)
    tokens = enc.encode(text)
    step = size - overlap
    spans = []
    for start_tok in range(0, len(tokens), step):
        end_tok = min(start_tok + size, len(tokens))
        # Convert token positions back into character positions.
        start_char = len(enc.decode(tokens[:start_tok]))
        end_char = len(enc.decode(tokens[:end_tok]))
        spans.append((start_char, end_char))
        if end_tok == len(tokens):
            break
    return spans


# ---------------------------------------------------------------------- #
# Strategy 2: recursive splitting on natural boundaries
# ---------------------------------------------------------------------- #
def _split_recursive(text: str, offset: int, separators: list[str], size: int,
                     model: str) -> list[tuple[int, int]]:
    """Break `text` into (start, end) pieces of at most `size` tokens each."""
    if count_tokens(text, model) <= size:
        return [(offset, offset + len(text))]
    if not separators:
        # No natural boundary left (one giant "word"): fall back to a hard cut.
        return [(offset + s, offset + e) for s, e in _fixed_spans(text, size, 0, model)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:  # this separator does not occur — try the next, weaker one
        return _split_recursive(text, offset, rest, size, model)

    pieces, pos = [], offset
    for i, part in enumerate(parts):
        part_with_sep = part + (sep if i < len(parts) - 1 else "")  # keep separators: no text is lost
        if part_with_sep:
            pieces.extend(_split_recursive(part_with_sep, pos, rest, size, model))
        pos += len(part_with_sep)
    return pieces


def _merge_pieces(text: str, pieces: list[tuple[int, int]], size: int, overlap: int,
                  model: str) -> list[tuple[int, int]]:
    """Glue small pieces into chunks of up to `size` tokens, with `overlap` tokens repeated."""
    piece_tokens = [count_tokens(text[s:e], model) for s, e in pieces]
    spans: list[tuple[int, int]] = []
    current: list[int] = []  # indexes into `pieces`
    current_tokens = 0

    for i, n in enumerate(piece_tokens):
        if current and current_tokens + n > size:
            spans.append((pieces[current[0]][0], pieces[current[-1]][1]))
            # Start the next chunk with the tail of this one (the overlap).
            kept, kept_tokens = [], 0
            for j in reversed(current):
                if kept_tokens + piece_tokens[j] > overlap:
                    break
                kept.insert(0, j)
                kept_tokens += piece_tokens[j]
            # The overlap + new piece must still fit; drop overlap pieces if not.
            while kept and kept_tokens + n > size:
                kept_tokens -= piece_tokens[kept.pop(0)]
            current, current_tokens = kept, kept_tokens
        current.append(i)
        current_tokens += n

    if current:
        spans.append((pieces[current[0]][0], pieces[current[-1]][1]))
    return spans


# ---------------------------------------------------------------------- #
# Public entry point
# ---------------------------------------------------------------------- #
def chunk_document(doc: Document, text: str, cfg: RAGConfig) -> list[Chunk]:
    log.stage(
        "I3", "CHUNKING",
        "Split the text into small overlapping pieces. Chunks are what we search for and what the LLM reads.",
    )
    model = cfg.embedding_model
    log.kv("Chunking settings", {
        "chunk_strategy": cfg.chunk_strategy,
        "chunk_size": f"{cfg.chunk_size} tokens (max per chunk)",
        "chunk_overlap": f"{cfg.chunk_overlap} tokens",
        "prepend_section_title": cfg.prepend_section_title,
    })

    if cfg.chunk_strategy == "fixed":
        log.explain(f"FIXED: a window of {cfg.chunk_size} tokens slides forward by "
                    f"{cfg.chunk_size - cfg.chunk_overlap} tokens each time, ignoring sentence boundaries.")
        spans = _fixed_spans(text, cfg.chunk_size, cfg.chunk_overlap, model)
    else:
        # Pieces must be no bigger than the overlap, otherwise no whole piece fits in the
        # overlap window and neighbouring chunks would share nothing.
        piece_size = cfg.chunk_overlap if cfg.chunk_overlap > 0 else cfg.chunk_size
        log.explain("RECURSIVE: split on paragraphs, then lines, then sentences, then words — only "
                    f"as deep as needed to get pieces of <= {piece_size} tokens — then merge neighbours "
                    "back up to chunk_size.")
        pieces = _split_recursive(text, 0, SEPARATORS, piece_size, model)
        log.detail(f"Recursive split produced {len(pieces)} small pieces (max {piece_size} tokens each) "
                   "before merging")
        spans = _merge_pieces(text, pieces, cfg.chunk_size, cfg.chunk_overlap, model)

    positions, paths = _section_index(text)
    log.detail(f"Found {len(positions)} markdown headings to use as section labels")

    chunks: list[Chunk] = []
    for start, end in spans:
        # Trim whitespace at the edges but keep exact character positions.
        raw = text[start:end]
        start += len(raw) - len(raw.lstrip())
        end -= len(raw) - len(raw.rstrip())
        if start >= end:
            continue
        chunk_text = text[start:end]
        # Label the chunk with the section active at its MIDDLE: the start may be overlap
        # text carried over from the previous section.
        chunks.append(Chunk(
            id=len(chunks),
            text=chunk_text,
            start_char=start,
            end_char=end,
            n_tokens=count_tokens(chunk_text, model),
            section=_section_at((start + end) // 2, positions, paths),
            source=doc.name,
        ))

    _log_chunk_report(chunks, text, cfg)
    return chunks


def _log_chunk_report(chunks: list[Chunk], text: str, cfg: RAGConfig) -> None:
    sizes = [c.n_tokens for c in chunks]
    log.step(f"Created [bold]{len(chunks)}[/] chunks")
    log.kv("Chunk size statistics (tokens)", {
        "min": min(sizes),
        "mean": f"{statistics.mean(sizes):.1f}",
        "median": statistics.median(sizes),
        "max": max(sizes),
        "total tokens in all chunks": f"{sum(sizes):,}",
        "document tokens": f"{count_tokens(text, cfg.embedding_model):,}",
        "duplication from overlap": f"{sum(sizes) / max(count_tokens(text, cfg.embedding_model), 1) - 1:+.1%}",
    })
    log.explain("'duplication from overlap' = extra tokens we embed because overlapping text is stored twice.")

    table = Table(title="Chunk overview", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("chars", justify="right", style="grey62")
    table.add_column("section", style="cyan", max_width=34, no_wrap=True)
    table.add_column("preview", style="grey70", no_wrap=True, max_width=70)
    for c in chunks:
        table.add_row(str(c.id), str(c.n_tokens), f"{c.start_char}-{c.end_char}",
                      esc(c.section), esc(c.preview()))
    log.table(table, level=2)

    # Show the overlap between the first two chunks, so it is visible.
    if len(chunks) > 1 and cfg.chunk_overlap > 0:
        a, b = chunks[0], chunks[1]
        if b.start_char < a.end_char:
            shared = text[b.start_char:a.end_char]
            log.block(f"Overlap between chunk 0 and chunk 1 ({count_tokens(shared)} tokens shared)",
                      shared, level=3, style="yellow")
        else:
            log.debug("Chunks 0 and 1 do not overlap (the boundary fell on a big paragraph break).")

    if chunks:
        c = chunks[0]
        log.block(f"Chunk 0 exactly as it will be embedded (prepend_section_title={cfg.prepend_section_title})",
                  c.embed_text(cfg.prepend_section_title), level=3, style="cyan")
    log.success(f"Chunking done — try `/set chunk_size 80` or `/set chunk_strategy fixed` and compare!")
