# RAG Pipeline: Flow Diagram

The bot works in two phases:

1. **Indexing** runs once at start-up. It turns the document into something searchable.
2. **Querying** runs for every question. It finds the relevant chunks and lets the LLM answer from them.

Every box carries a step code (`I1`… / `Q1`…). The same code appears as a header in the terminal logs, so you can match what you see on screen to this diagram and to the file that does the work.

> The Mermaid diagrams below render on GitHub and in VS Code (with a Mermaid extension).
> The same picture in plain text is printed in the terminal at start-up, and again whenever you type `/diagram`.

---

## 1. Overview

```mermaid
flowchart LR
    subgraph INDEXING["INDEXING (once)"]
        direction TB
        A[(Document<br/>data/*.md / .txt / .pdf)] --> I1[I1 Load] --> I2[I2 Clean] --> I3[I3 Chunk]
        I3 --> I4[I4 Embed chunks<br/>OpenAI embeddings]
        I4 --> I5[(I5 Vector store<br/>n x 1536 matrix)]
        I3 --> I6[(I6 BM25 index<br/>terms, df, IDF)]
    end

    subgraph QUERY["QUERYING (every question)"]
        direction TB
        Q1[/Q1 Question/] --> Q2[Q2 Rewrite query<br/>OpenAI chat + history]
        Q2 --> Q3[Q3 Embed question<br/>OpenAI embeddings]
        Q3 --> Q4[Q4 Vector search<br/>cosine similarity]
        Q2 --> Q5[Q5 BM25 search<br/>keyword scoring]
        Q4 --> Q6{Q6 Hybrid fusion<br/>RRF or weighted}
        Q5 --> Q6
        Q6 --> Q7[Q7 Rerank<br/>optional, OpenAI chat]
        Q7 --> Q8[Q8 Filter<br/>threshold + top_k]
        Q8 --> Q9[Q9 Build prompt<br/>rules + history + chunks]
        Q9 --> Q10[Q10 Generate<br/>OpenAI chat, streamed]
        Q10 --> Q11[/Q11 Answer + sources + cost/]
    end

    I5 -.-> Q4
    I6 -.-> Q5
```

---

## 2. Indexing phase in detail

```mermaid
flowchart TD
    F[(nimbus_handbook.md)] --> L

    L["<b>I1 LOAD</b> · loader.py<br/>check path is inside workspace<br/>read text / extract PDF pages<br/>log size, chars, tokens"]
    L --> C["<b>I2 CLEAN</b> · cleaner.py<br/>NFKC unicode · \r\n → \n<br/>trailing / repeated spaces · max 1 blank line"]
    C --> K["<b>I3 CHUNK</b> · chunker.py<br/>strategy: recursive | fixed<br/>chunk_size, chunk_overlap (tokens)<br/>section path from markdown headings"]

    K -->|"chunk text<br/>(+ 'Section: …' prefix)"| E["<b>I4 EMBED</b> · embedder.py<br/>check disk cache (.rag_cache)<br/>batch MISSES → OpenAI /embeddings<br/>text-embedding-3-small → 1536 floats"]
    E --> V[("<b>I5 VECTOR STORE</b> · vector_store.py<br/>row-normalised numpy matrix")]

    K -->|"section + chunk text"| T["text_tokenizer.py<br/>lowercase → words → drop stopwords → stem"]
    T --> B[("<b>I6 BM25 INDEX</b> · bm25.py<br/>tf per chunk · df per term<br/>IDF = ln((N-df+0.5)/(df+0.5)+1)<br/>avgdl")]
```

---

## 3. Query phase in detail

```mermaid
flowchart TD
    U[/"You: 'and its wind limit?'"/] --> Q1["<b>Q1</b> receive question"]
    Q1 --> H{chat history<br/>empty?}
    H -- yes --> SQ[search query = question]
    H -- no --> Q2["<b>Q2 REWRITE</b> · query_rewriter.py<br/>LLM → 'What is the max wind speed<br/>of the NX-90 Sparrow drone?'"]
    Q2 --> SQ

    SQ --> M{retrieval_mode}
    M -- "vector / hybrid" --> Q3["<b>Q3</b> embed search query"]
    Q3 --> Q4["<b>Q4 VECTOR SEARCH</b><br/>scores = matrix @ q<br/>top candidates_per_retriever"]
    M -- "bm25 / hybrid" --> Q5["<b>Q5 BM25 SEARCH</b><br/>Σ IDF · tf·(k1+1) / (tf + k1·(1-b+b·|D|/avgdl))"]

    Q4 --> Q6{"<b>Q6 FUSION</b> (hybrid only)<br/>RRF: Σ 1/(k+rank)<br/>weighted: α·norm(cos) + (1-α)·norm(bm25)"}
    Q5 --> Q6
    Q6 --> R{use_reranker?}
    R -- yes --> Q7["<b>Q7 RERANK</b> · reranker.py<br/>LLM scores each candidate 0-10"]
    R -- no --> Q8
    Q7 --> Q8["<b>Q8 FILTER</b> · retriever.py<br/>drop cosine &lt; similarity_threshold<br/>keep top_k"]

    Q8 --> Q9["<b>Q9 PROMPT</b> · prompt_builder.py<br/>system rules (answer only from context, cite [C#])<br/>+ last history_turns turns<br/>+ chunks within max_context_tokens<br/>+ ORIGINAL question"]
    Q9 --> Q10["<b>Q10 GENERATE</b> · generator.py<br/>chat.completions (stream=True)<br/>temperature, max_answer_tokens"]
    Q10 --> Q11[/"<b>Q11</b> answer · cited chunks · tokens · cost"/]
    Q11 -->|"saved as a turn"| HIST[(chat history)]
    HIST -.-> H
```

---

## 4. Which file does what

| Step | What happens | File | Settings that change it |
|---|---|---|---|
| I1 | Load the file (workspace only) | `rag/ingestion/loader.py` | `document_path` |
| I2 | Normalise text | `rag/ingestion/cleaner.py` | (none) |
| I3 | Split into chunks | `rag/ingestion/chunker.py` | `chunk_strategy`, `chunk_size`, `chunk_overlap`, `prepend_section_title` |
| I4 / Q3 | Text → vector | `rag/embeddings/embedder.py`, `cache.py` | `embedding_model`, `use_embedding_cache` |
| I5 / Q4 | Cosine similarity search | `rag/retrieval/vector_store.py`, `similarity.py` | `candidates_per_retriever` |
| I6 / Q5 | BM25 keyword search | `rag/retrieval/bm25.py`, `text_tokenizer.py` | `bm25_k1`, `bm25_b`, `bm25_remove_stopwords`, `bm25_use_stemming` |
| Q2 | Rewrite follow-up questions | `rag/generation/query_rewriter.py` | `use_query_rewriting`, `history_turns` |
| Q6 | Merge both rankings | `rag/retrieval/hybrid.py` | `retrieval_mode`, `fusion_method`, `hybrid_alpha`, `rrf_k` |
| Q7 | LLM reranking | `rag/retrieval/reranker.py` | `use_reranker` |
| Q8 | Threshold + top_k | `rag/retrieval/retriever.py` | `similarity_threshold`, `top_k` |
| Q9 | Build the prompt | `rag/generation/prompt_builder.py` | `max_context_tokens`, `history_turns` |
| Q10 | Stream the answer | `rag/generation/generator.py` | `chat_model`, `temperature`, `max_answer_tokens` |
| Q11 | Sources, tokens, cost | `rag/pipeline.py` | (none) |
