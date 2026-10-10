"""
================================================================================
 KEYWORD TOKENIZER  —  how BM25 turns text into searchable "terms"
================================================================================

NOTE: this is NOT the same as LLM tokens (tiktoken). BM25 works on whole words.

Pipeline for every chunk AND every question (must be identical for both!):

    "The Sparrow drones' batteries are charging..."
      1. lowercase        -> "the sparrow drones' batteries are charging..."
      2. split into words -> ["the", "sparrow", "drones", "batteries", "are", "charging"]
      3. drop stopwords   -> ["sparrow", "drones", "batteries", "charging"]
      4. stem             -> ["sparrow", "drone", "battery", "charg"]

Notice the crude stemmer maps "charging" -> "charg" but leaves "charge" alone,
so those two will NOT match. Stemming is always a trade-off.

Codes like "E-4471" or "NX-220" are kept as ONE term ("e-4471"), because
exact codes are precisely where keyword search beats embeddings.

The stemmer here is a tiny hand-written suffix stripper so you can read it.
Real systems use Porter/Snowball stemmers or lemmatisers (e.g. NLTK, spaCy).
"""

from __future__ import annotations

import re

# Words so common they carry almost no meaning for search.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves get got also may must
""".split())

# Word = letters/digits, optionally joined by "-" or "." (keeps "e-4471", "v4.7.2", "nx-220").
WORD_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

def simple_stem(word: str) -> str:
    """Strip common English suffixes. First matching rule wins."""
    if any(ch.isdigit() for ch in word):                 # never stem codes like "e-4471"
        return word
    if word.endswith("ies") and len(word) > 4:           # policies -> policy
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")) and len(word) > 4:
        return word[:-2]                                 # watches -> watch, boxes -> box
    if word.endswith("ing") and len(word) > 6:           # charging -> charg
        return word[:-3]
    if word.endswith("ed") and len(word) > 5:            # charged -> charg
        return word[:-2]
    if word.endswith("ly") and len(word) > 5:            # quickly -> quick
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]                                 # drones -> drone, days -> day
    return word


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    words = WORD_RE.findall(text.lower())
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    if stem:
        words = [simple_stem(w) for w in words]
    return words
