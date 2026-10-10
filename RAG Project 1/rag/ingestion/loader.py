"""
================================================================================
 INDEXING STEP I1  —  LOAD THE DOCUMENT
================================================================================

Goal: turn a file on disk into one big Python string + some metadata.

Safety rule: the bot only ingests files INSIDE this project folder (the
"workspace"). A path like "../../Windows/secret.txt" is rejected. In real
products, controlling *what* gets ingested is a security concern: whatever is
in the index can end up in an answer.

Supported formats:
    .txt / .md   -> read as UTF-8 text
    .pdf         -> text extracted page by page with `pypdf`
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import PROJECT_ROOT
from rag.logger import esc, log
from rag.utils.tokens import count_tokens

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    path: Path                     # absolute path on disk
    text: str                      # the full text content
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


def resolve_workspace_path(path_str: str) -> Path:
    """Resolve `path_str` and make sure it is inside the project folder."""
    path = (PROJECT_ROOT / path_str).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise PermissionError(
            f"'{path_str}' is outside the workspace ({PROJECT_ROOT}). "
            "Only files inside this project can be ingested."
        )
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use one of {sorted(SUPPORTED_EXTENSIONS)}")
    return path


def _read_pdf(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    for i, page_text in enumerate(pages, start=1):
        log.debug(f"page {i}: {len(page_text):,} characters extracted")
    return "\n\n".join(pages), len(pages)


def load_document(path_str: str) -> Document:
    log.stage(
        "I1", "LOAD DOCUMENT",
        "Read the raw file from disk into memory. Everything after this step works on plain text.",
    )

    path = resolve_workspace_path(path_str)
    log.step(f"Path resolved to [bold]{esc(str(path))}[/]")
    log.explain("We check the path is inside the project folder, so the bot can only learn from workspace files.")

    raw_bytes = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        text, pages = _read_pdf(path)
    else:
        text, pages = raw_bytes.decode("utf-8", errors="replace"), 1

    metadata: dict[str, object] = {
        "file name": path.name,
        "format": path.suffix.lower(),
        "size on disk": f"{len(raw_bytes):,} bytes",
        "pages": pages,
        "characters": f"{len(text):,}",
        "words (approx.)": f"{len(text.split()):,}",
        "lines": f"{text.count(chr(10)) + 1:,}",
        "tokens (cl100k)": f"{count_tokens(text):,}",
        "sha256": hashlib.sha256(raw_bytes).hexdigest()[:16] + "...",
    }
    log.kv("Document statistics", metadata)
    log.explain(
        "The whole document is far too large to paste into every prompt (cost + context limits), "
        "which is exactly why we will chunk it and only retrieve the relevant pieces."
    )
    log.block("First 400 characters of the raw file", text[:400], level=3)

    log.success(f"Loaded '{esc(path.name)}'")
    return Document(path=path, text=text, metadata=metadata)
