"""
================================================================================
 THE RAG PIPELINE  —  the conductor that calls every step in order
================================================================================

Read this file first: it is the "table of contents" of the whole project.
Every step lives in its own module; this class just wires them together.

 INDEXING (once per document, or again after `/set chunk_size ...`)
   I1 load_document      rag/ingestion/loader.py
   I2 clean_text         rag/ingestion/cleaner.py
   I3 chunk_document     rag/ingestion/chunker.py
   I4 embed chunks       rag/embeddings/embedder.py
   I5 VectorStore        rag/retrieval/vector_store.py
   I6 BM25Index          rag/retrieval/bm25.py

 QUERYING (every question)
   Q1  receive question  (this file)
   Q2  rewrite_query     rag/generation/query_rewriter.py
   Q3  embed question    rag/embeddings/embedder.py
   Q4  vector search     rag/retrieval/vector_store.py
   Q5  BM25 search       rag/retrieval/bm25.py
   Q6  hybrid fusion     rag/retrieval/hybrid.py
   Q7  rerank (opt.)     rag/retrieval/reranker.py
   Q8  filter + top_k    rag/retrieval/retriever.py
   Q9  build prompt      rag/generation/prompt_builder.py
   Q10 generate answer   rag/generation/generator.py
   Q11 sources + stats   (this file)
"""

from __future__ import annotations

import time

from rag.config import RAGConfig
from rag.embeddings.embedder import describe_vector, embed_texts
from rag.generation.generator import cited_chunk_ids, generate_answer
from rag.generation.prompt_builder import build_messages
from rag.generation.query_rewriter import rewrite_query
from rag.ingestion.chunker import Chunk, chunk_document
from rag.ingestion.cleaner import clean_text
from rag.ingestion.loader import Document, load_document
from rag.logger import Table, esc, log
from rag.memory.chat_history import ChatHistory
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.hybrid import Candidate
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector_store import VectorStore
from rag.utils.usage import format_cost, usage


class RAGPipeline:
    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.history = ChatHistory()
        self.document: Document | None = None
        self.chunks: list[Chunk] = []
        self.bm25: BM25Index | None = None
        self.retriever: Retriever | None = None

    # ================================================================== #
    # INDEXING
    # ================================================================== #
    def build_index(self) -> None:
        cfg = self.cfg
        log.banner("INDEXING PHASE", f"Preparing '{cfg.document_path}' so it can be searched")
        start = time.perf_counter()
        calls_before = len(usage.calls)

        self.document = load_document(cfg.document_path)                 # I1
        text = clean_text(self.document.text)                              # I2
        self.chunks = chunk_document(self.document, text, cfg)             # I3

        log.stage("I4", "EMBED ALL CHUNKS",
                  "Turn every chunk into a vector of numbers that captures its meaning.")
        log.detail(f"Embedding model: {cfg.embedding_model}")
        with log.timer("Embedding all chunks"):
            vectors = embed_texts([c.embed_text(cfg.prepend_section_title) for c in self.chunks],
                                  cfg, purpose="embed chunks")                 # I4
        describe_vector("Chunk 0 vector", vectors[0])

        vector_store = VectorStore(self.chunks, vectors)                    # I5
        self.bm25 = BM25Index(self.chunks, cfg.bm25_k1, cfg.bm25_b,          # I6
                              cfg.bm25_remove_stopwords, cfg.bm25_use_stemming)
        self.retriever = Retriever(self.chunks, vector_store, self.bm25)

        cost = sum(c.cost or 0 for c in usage.calls[calls_before:])
        log.always("")
        log.success(f"[bold]Index ready:[/] {len(self.chunks)} chunks from '{esc(self.document.name)}' "
                    f"in {time.perf_counter() - start:.1f}s (indexing cost ${cost:.6f})")

    # ================================================================== #
    # QUERYING
    # ================================================================== #
    def ask(self, question: str) -> str:
        assert self.retriever is not None and self.document is not None, "call build_index() first"
        cfg = self.cfg
        log.banner("QUERY PHASE", f"Question: {question}")
        start = time.perf_counter()
        calls_before = len(usage.calls)

        log.stage("Q1", "QUESTION RECEIVED", "The user's raw question enters the pipeline.")
        log.step(f"Question: [bold]{esc(question)}[/]")
        log.detail(f"Chat history: {len(self.history)} previous turn(s)")

        search_query = rewrite_query(question, self.history, cfg)                         # Q2
        selected = self.retriever.retrieve(search_query, cfg)                              # Q3-Q8
        messages, used_ids = build_messages(question, selected, self.chunks, self.history,  # Q9
                                            cfg, self.document.name)
        answer = generate_answer(messages, cfg)                                            # Q10
        self._report(answer, used_ids, selected, start, calls_before)                      # Q11

        self.history.add(question, answer)
        return answer

    def search_only(self, query: str) -> list[Candidate]:
        """Run retrieval WITHOUT calling the LLM (the /search command)."""
        assert self.retriever is not None
        log.banner("RETRIEVAL ONLY", f"Query: {query}")
        return self.retriever.retrieve(query, self.cfg)

    # ------------------------------------------------------------------ #
    def _report(self, answer: str, used_ids: list[int], selected: list[Candidate],
                start: float, calls_before: int) -> None:
        log.stage("Q11", "SOURCES & STATS", "Which chunks did the answer actually use, and what did it cost?")
        cited = cited_chunk_ids(answer)
        unknown = [c for c in cited if c not in used_ids]

        if not used_ids:
            log.step("No chunks were given to the LLM, so there are no sources.")
        table = Table(title="Sources", title_justify="left")
        table.add_column("chunk", justify="right")
        table.add_column("cited?", justify="center")
        table.add_column("section", style="cyan", max_width=36, no_wrap=True)
        table.add_column("characters in document", style="grey62")
        for cid in used_ids:
            ch = self.chunks[cid]
            mark = "[bold green]YES[/]" if cid in cited else "[grey50]no[/]"
            table.add_row(f"C{cid}", mark, esc(ch.section), f"{ch.start_char}-{ch.end_char}")
        if used_ids:
            log.table(table, level=1)
        if used_ids and not cited:
            log.detail("The answer cites no chunk — usually it means the context did not contain the answer.")
        if unknown:
            log.warn(f"The answer cites chunk(s) {unknown} that were NOT in the prompt — a hallucinated citation!")
        if any(cid not in cited for cid in used_ids):
            log.explain("Chunks retrieved but not cited were noise for this question: lowering top_k or "
                        "raising similarity_threshold would remove them (and save tokens).")

        calls = usage.calls[calls_before:]
        t = Table(title="API calls for this question", title_justify="left")
        for col in ("purpose", "model", "in tokens", "out tokens", "ms", "cost"):
            t.add_column(col, justify="right" if col not in ("purpose", "model") else "left")
        for c in calls:
            t.add_row(c.purpose, c.model, f"{c.input_tokens:,}", f"{c.output_tokens:,}",
                      f"{c.seconds * 1000:.0f}", format_cost(c.cost))
        log.table(t, level=1)
        total = sum(c.cost or 0 for c in calls)
        log.step(f"Answered in {time.perf_counter() - start:.2f}s, {len(calls)} API call(s), "
                 f"~${total:.6f}  (session total ~${usage.total_cost:.6f})")
