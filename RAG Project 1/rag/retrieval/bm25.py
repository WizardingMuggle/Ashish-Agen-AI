"""
================================================================================
 INDEXING STEP I6 (and QUERY STEP Q5)  —  BM25 KEYWORD SEARCH (from scratch)
================================================================================

BM25 ("Best Match 25") is the classic search-engine ranking formula, used by
Elasticsearch, Lucene, OpenSearch... It needs no AI and no API calls.

For a query Q with terms q1..qn and a chunk D:

    score(D, Q) = SUM over terms q in Q of

                                 tf(q, D) * (k1 + 1)
        IDF(q)  *  ------------------------------------------------
                    tf(q, D) + k1 * (1 - b + b * |D| / avgdl)

  tf(q, D)  term frequency: how many times q appears in chunk D
  |D|       length of chunk D (in terms);  avgdl = average chunk length
  IDF(q)    inverse document frequency: how RARE the term is

                 N - df(q) + 0.5
    IDF = ln(  ----------------- + 1 )      N = number of chunks
                   df(q) + 0.5               df = number of chunks containing q

Intuition:
  * Rare words matter more: "e-4471" (in 1 chunk) >> "robot" (in 20 chunks).
  * Saturation (k1): the 1st occurrence of a word counts a lot, the 10th barely.
  * Length normalisation (b): a word in a short chunk is more significant than
    the same word buried in a long one.

Strength:  exact matches — codes, names, IDs, rare jargon ("E-4471", "Voyager").
Weakness:  no understanding of meaning — "vacation" does NOT match "annual leave".
That is why we combine BM25 with vector search (hybrid retrieval).
"""

from __future__ import annotations

import math
from collections import Counter

from rag.ingestion.chunker import Chunk
from rag.logger import Table, esc, log
from rag.retrieval.text_tokenizer import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75,
                 remove_stopwords: bool = True, stem: bool = True) -> None:
        log.stage(
            "I6", "BUILD BM25 KEYWORD INDEX",
            "Tokenise every chunk into terms and pre-compute term frequencies and IDF for keyword scoring.",
        )
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.remove_stopwords, self.stem = remove_stopwords, stem

        # 1) Tokenise every chunk (section titles included, like for embeddings).
        self.doc_terms: list[list[str]] = [self._tokenize(f"{c.section}\n{c.text}") for c in chunks]
        self.term_freqs: list[Counter[str]] = [Counter(terms) for terms in self.doc_terms]
        self.doc_lengths = [len(terms) for terms in self.doc_terms]
        self.N = len(chunks)
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        # 2) Document frequency: in how many chunks does each term appear?
        self.df: Counter[str] = Counter()
        for tf in self.term_freqs:
            self.df.update(tf.keys())

        # 3) IDF for every term in the vocabulary.
        self.idf = {term: self._idf(df) for term, df in self.df.items()}

        self._log_build()

    # ------------------------------------------------------------------ #
    def _tokenize(self, text: str) -> list[str]:
        return tokenize(text, self.remove_stopwords, self.stem)

    def _idf(self, df: int) -> float:
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _term_score(self, term: str, doc_idx: int) -> float:
        tf = self.term_freqs[doc_idx].get(term, 0)
        if tf == 0:
            return 0.0
        length_norm = 1 - self.b + self.b * self.doc_lengths[doc_idx] / self.avgdl
        return self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        return sum(self._term_score(t, doc_idx) for t in query_terms if t in self.idf)

    def all_scores(self, query: str) -> list[float]:
        """BM25 score of EVERY chunk for this query (used by weighted fusion)."""
        terms = self._tokenize(query)
        return [self.score(terms, i) for i in range(self.N)]

    # ------------------------------------------------------------------ #
    def _log_build(self) -> None:
        log.kv("BM25 index", {
            "chunks (N)": self.N,
            "vocabulary size": f"{len(self.df):,} unique terms",
            "average chunk length (avgdl)": f"{self.avgdl:.1f} terms",
            "k1 (term-frequency saturation)": self.k1,
            "b (length normalisation)": self.b,
            "stopword removal": self.remove_stopwords,
            "stemming": self.stem,
        })
        if self.chunks:
            sample = self.chunks[0].text[:160]
            log.debug(f"Tokeniser example (chunk 0): {esc(repr(sample))}")
            log.debug(f"  -> {esc(str(self._tokenize(sample)))}")

        by_idf = sorted(self.idf.items(), key=lambda kv: kv[1])
        common = ", ".join(f"{t}({self.df[t]})" for t, _ in by_idf[:10])
        rare = ", ".join(f"{t}({self.df[t]})" for t, _ in by_idf[-10:])
        log.detail(f"Most common terms -> LOW IDF  (term(df)): {esc(common)}")
        log.detail(f"Rarest terms      -> HIGH IDF (term(df)): {esc(rare)}")
        log.explain("A term found in almost every chunk tells us nothing (low IDF); a term found in one "
                    "chunk is a strong signal (high IDF).")
        log.success("BM25 index ready")

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        log.stage(
            "Q5", "BM25 KEYWORD SEARCH",
            "Score every chunk by how many RARE query words it contains (exact word matching, no meaning).",
        )
        terms = self._tokenize(query)
        log.step(f"Query terms after tokenising: [bold]{esc(str(terms))}[/]")
        if not terms:
            log.warn("No searchable terms left in the query (all stopwords?) — BM25 returns nothing.")
            return []

        table = Table(title="Query terms", title_justify="left")
        table.add_column("term", style="bold")
        table.add_column("df (chunks containing it)", justify="right")
        table.add_column("IDF", justify="right", style="green")
        for t in dict.fromkeys(terms):
            if t in self.idf:
                table.add_row(esc(t), str(self.df[t]), f"{self.idf[t]:.3f}")
            else:
                table.add_row(esc(t), "0", "[red]not in vocabulary — ignored[/]")
        log.table(table)

        scores = [(i, self.score(terms, i)) for i in range(self.N)]
        matched = sorted([s for s in scores if s[1] > 0], key=lambda s: -s[1])
        results = matched[:k]
        log.step(f"{len(matched)} of {self.N} chunks contain at least one query term; keeping top {len(results)}")

        if results:
            log.bar_chart(f"Top {len(results)} chunks by BM25 score",
                          [(f"chunk {i}", s) for i, s in results])
            self._log_breakdown(terms, results)
        log.explain("BM25 scores are NOT on a 0-1 scale; they grow with the number and rarity of matching "
                    "terms. Compare them only within the same query.")
        return results

    def _log_breakdown(self, terms: list[str], results: list[tuple[int, float]]) -> None:
        """Show WHERE each chunk's score comes from, term by term."""
        unique_terms = [t for t in dict.fromkeys(terms) if t in self.idf]
        table = Table(title="BM25 score breakdown (contribution of each term)", title_justify="left")
        table.add_column("chunk", justify="right")
        table.add_column("len", justify="right", style="grey62")
        for t in unique_terms:
            table.add_column(esc(t), justify="right")
        table.add_column("TOTAL", justify="right", style="bold green")
        for i, total in results:
            cells = []
            for t in unique_terms:
                tf = self.term_freqs[i].get(t, 0)
                cells.append(f"{self._term_score(t, i):.2f} (tf={tf})" if tf else "[grey50]-[/]")
            table.add_row(str(i), str(self.doc_lengths[i]), *cells, f"{total:.3f}")
        log.table(table, level=3)

    # ------------------------------------------------------------------ #
    def describe_term(self, word: str) -> None:
        """Used by the /term command: everything BM25 knows about one word."""
        term_list = self._tokenize(word)
        if not term_list:
            log.always(f"'{esc(word)}' is a stopword (or empty) — BM25 ignores it completely.")
            return
        term = term_list[0]
        if term not in self.df:
            log.always(f"'{esc(word)}' -> term '{esc(term)}' does not appear in any chunk.")
            return
        chunk_ids = [i for i, tf in enumerate(self.term_freqs) if term in tf]
        log.always(f"'{esc(word)}' -> term '[bold]{esc(term)}[/]': appears in {self.df[term]}/{self.N} chunks, "
                   f"IDF = {self.idf[term]:.3f}")
        log.always(f"  chunks: {', '.join(f'{i}(tf={self.term_freqs[i][term]})' for i in chunk_ids)}")
