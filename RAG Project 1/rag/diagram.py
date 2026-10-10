"""The pipeline flow diagram printed at start-up and by the /diagram command."""

FLOW_DIAGRAM = r"""
 ============================ INDEXING PHASE (runs once at start-up) ============================

   data/nimbus_handbook.md
            |
            v
   [I1] LOAD ----------> [I2] CLEAN ----------> [I3] CHUNK -----------------------------+
   loader.py             cleaner.py             chunker.py                              |
   read file, check      normalise unicode,     recursive / fixed split                 |
   it is in workspace    whitespace, blanks     chunk_size + chunk_overlap (tokens)     |
                                                                                        |
                         +--------------------------------------------------------------+
                         |                                            |
                         v                                            v
              [I4] EMBED CHUNKS (OpenAI)                   [I6] BUILD BM25 INDEX
              embedder.py + cache.py                       bm25.py + text_tokenizer.py
              text -> 1536-dim vector                      tokenise, stopwords, stem,
              cached in .rag_cache/                        term freq, doc freq, IDF
                         |                                            |
                         v                                            |
              [I5] VECTOR STORE                                       |
              vector_store.py: matrix (n x 1536)                      |
                         |                                            |
 ========================|============ QUERY PHASE (every question) ==|==========================
                         |                                            |
   You: "and its wind limit?"                                         |
            |                                                         |
            v                                                         |
   [Q1] QUESTION --> [Q2] REWRITE (OpenAI chat, uses history)         |
                     "What is the max wind speed of the Sparrow?"     |
                          |                    |                      |
                          v                    v                      |
           [Q3] EMBED QUESTION (OpenAI)   [Q5] BM25 SEARCH <----------+
                          |               keyword scores
                          v                    |
           [Q4] VECTOR SEARCH <-- I5           |
           cosine similarity                   |
                          |                    |
                          +---------+----------+
                                    v
                       [Q6] HYBRID FUSION (hybrid.py)
                       RRF: 1/(k+rank)  or  alpha*vec + (1-alpha)*bm25
                                    |
                                    v
                       [Q7] RERANK (optional, OpenAI chat judges 0-10)
                                    |
                                    v
                       [Q8] FILTER: similarity_threshold, keep top_k
                                    |
                                    v
                       [Q9] BUILD PROMPT: system rules + history + [C#] chunks + question
                                    |
                                    v
                       [Q10] GENERATE (OpenAI chat, streamed)
                                    |
                                    v
                       [Q11] ANSWER + cited sources + tokens + cost
"""
