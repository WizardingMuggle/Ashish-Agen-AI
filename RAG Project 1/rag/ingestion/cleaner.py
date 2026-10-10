"""
================================================================================
 INDEXING STEP I2  —  CLEAN / NORMALISE THE TEXT
================================================================================

Raw text is messy: Windows line endings (\r\n), tabs, trailing spaces, runs of
blank lines, "smart" unicode characters from PDFs (ligatures like "ﬁ")...

Why clean?
  * Garbage whitespace wastes tokens (= money) in every chunk and prompt.
  * Inconsistent characters hurt keyword search: "ﬁle" (ligature) != "file".
  * Our chunker splits on blank lines, so they must be consistent.

Each rule below is applied in order and the logs show how many characters
each one changed — try ingesting a PDF to see much bigger numbers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

from rag.logger import Table, log

# (rule name, explanation, function)
CLEANING_RULES: list[tuple[str, str, Callable[[str], str]]] = [
    ("unicode NFKC", "turn look-alike unicode (ligatures, full-width chars) into plain forms",
     lambda t: unicodedata.normalize("NFKC", t)),
    ("line endings", "convert Windows \\r\\n and old Mac \\r into \\n",
     lambda t: t.replace("\r\n", "\n").replace("\r", "\n")),
    ("tabs -> space", "tabs become a single space",
     lambda t: t.replace("\t", " ")),
    ("trailing spaces", "remove spaces at the end of every line",
     lambda t: re.sub(r"[ ]+\n", "\n", t)),
    ("repeated spaces", "collapse runs of spaces inside a line",
     lambda t: re.sub(r"(?<=\S)[ ]{2,}", " ", t)),
    ("blank lines", "max ONE empty line between paragraphs",
     lambda t: re.sub(r"\n{3,}", "\n\n", t)),
    ("strip", "remove whitespace at the very start/end",
     lambda t: t.strip()),
]


def clean_text(text: str) -> str:
    log.stage(
        "I2", "CLEAN TEXT",
        "Normalise whitespace and characters so chunking and keyword search behave consistently.",
    )

    table = Table(title="Cleaning rules applied (in order)", title_justify="left")
    table.add_column("Rule", style="bold")
    table.add_column("What it does", style="grey70")
    table.add_column("Chars before", justify="right")
    table.add_column("Chars after", justify="right")
    table.add_column("Removed", justify="right", style="yellow")

    original_length = len(text)
    for name, description, rule in CLEANING_RULES:
        before = len(text)
        text = rule(text)
        table.add_row(name, description, f"{before:,}", f"{len(text):,}", f"{before - len(text):,}")
    log.table(table)

    removed = original_length - len(text)
    log.step(f"Cleaned text: {len(text):,} characters ({removed:,} removed, "
             f"{removed / max(original_length, 1):.1%} of the original)")
    log.explain("A small percentage is normal for a clean markdown file; scanned PDFs often lose 5-15%.")
    return text
