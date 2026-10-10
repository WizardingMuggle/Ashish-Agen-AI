# RAG Chatbot: Learn Retrieval-Augmented Generation by Watching It Work

A complete RAG chatbot that runs in the terminal and **explains everything it does as it does it**.
Every stage of the pipeline prints what it is doing, why it is doing it, and the actual numbers: the chunks, vectors, cosine scores, BM25 term weights, fusion ranks, the full prompt, token counts and cost.
Change a setting live with `/set`, ask the same question again, and watch the logs and the answer change.

Built with the **OpenAI API**: embeddings come from `text-embedding-3-small` and answers from `gpt-4.1-mini`. Both are configurable.

---

## 1. Quick start

```powershell
# 1. Create a virtual environment and install the dependencies
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your OpenAI key
copy .env.example .env            # macOS/Linux: cp .env.example .env
#    then open .env and paste your key after OPENAI_API_KEY=

# 3. Run
python main.py
```

Then ask: `How many vacation days do I get?`

Useful start-up options:

```powershell
python main.py --verbosity 1                 # quieter: stage headers + results only
python main.py --mode bm25                   # keyword search only (no embeddings at query time)
python main.py --file data/my_notes.pdf      # ingest another file from this workspace
```

> The very first run downloads tiktoken's tokenizer files (a few seconds), and embeds all chunks (well under $0.001).
> Embeddings are cached in `.rag_cache/`, so later runs start almost instantly.

---

## 2. What is RAG?

An LLM only knows what was in its training data. It knows nothing about **your** documents, such as a company handbook, product manuals or notes.

**Retrieval-Augmented Generation** fixes this without retraining the model:

1. **Retrieve**: find the few pieces of your document most relevant to the question.
2. **Augment**: paste those pieces into the prompt.
3. **Generate**: the LLM writes an answer *using only those pieces*, and cites them.

The sample document `data/nimbus_handbook.md` describes a **fictional** company. The model cannot know these facts from training, so every correct answer proves that retrieval worked.

---

## 3. The pipeline

See **[docs/FLOW_DIAGRAM.md](docs/FLOW_DIAGRAM.md)** for the full diagrams. You can also type `/diagram` in the chat.

```
INDEXING (once)   I1 Load → I2 Clean → I3 Chunk ─┬→ I4 Embed → I5 Vector store
                                                 └→ I6 BM25 index

QUERY (each Q)    Q1 Question → Q2 Rewrite ─┬→ Q3 Embed → Q4 Vector search ─┐
                                            └→ Q5 BM25 search ──────────────┴→ Q6 Fusion
                  → Q7 Rerank (opt.) → Q8 Filter/top_k → Q9 Prompt → Q10 Generate → Q11 Sources & cost
```

Each code (`[I3]`, `[Q6]`, …) appears as a header in the terminal logs.

---

## 4. Project structure

```
RAG Project 1/
├── main.py                        ← start here: the terminal chatbot and the /commands
├── requirements.txt
├── .env.example                   ← copy to .env and add your OPENAI_API_KEY
├── data/
│   └── nimbus_handbook.md         ← the sample document that gets ingested
├── docs/
│   ├── FLOW_DIAGRAM.md            ← Mermaid flow diagrams + file-to-step table
│   └── EXPERIMENTS.md             ← 13 guided labs: change X, observe Y
└── rag/
    ├── pipeline.py                ← THE CONDUCTOR: calls every step in order (read this first)
    ├── config.py                  ← every tunable setting, with explanations
    ├── logger.py                  ← the teaching logs (verbosity levels, tables, WHY notes)
    ├── diagram.py                 ← ASCII diagram printed in the terminal
    ├── ingestion/
    │   ├── loader.py              ← I1  read the file (workspace-only safety check)
    │   ├── cleaner.py             ← I2  normalise whitespace / unicode
    │   └── chunker.py             ← I3  recursive & fixed chunking with overlap
    ├── embeddings/
    │   ├── openai_client.py       ←     one shared OpenAI client
    │   ├── embedder.py            ← I4/Q3  text → vectors (batched)
    │   └── cache.py               ←     on-disk embedding cache
    ├── retrieval/
    │   ├── similarity.py          ←     cosine, dot product, euclidean
    │   ├── vector_store.py        ← I5/Q4  semantic search
    │   ├── text_tokenizer.py      ←     words, stopwords, stemming (for BM25)
    │   ├── bm25.py                ← I6/Q5  BM25 written from scratch
    │   ├── hybrid.py              ← Q6  RRF and weighted fusion
    │   ├── reranker.py            ← Q7  LLM-as-judge reranking
    │   └── retriever.py           ← Q3–Q8 glue: modes, threshold, top_k
    ├── generation/
    │   ├── query_rewriter.py      ← Q2  follow-up → standalone question
    │   ├── prompt_builder.py      ← Q9  system rules + history + context + token budget
    │   └── generator.py           ← Q10 streaming answer + citations
    ├── memory/
    │   └── chat_history.py        ←     conversation memory
    └── utils/
        ├── tokens.py              ←     tiktoken token counting
        └── usage.py               ←     API call / token / cost tracking
```

**Suggested reading order:** `rag/pipeline.py` → `rag/config.py` → then each step in order, I1 → Q11. Every file begins with a docstring that explains the concept before the code.

---

## 5. Chat commands

| Command | What it does |
|---|---|
| *any text* | Ask a question about the document |
| `/help` | List the commands |
| `/config` | Show all settings and their current values |
| `/set <key> <value>` | Change a setting live. Chunking/BM25 settings re-index automatically |
| `/verbosity <0-3>` | 0 = answers only · 1 = stages · 2 = + tables & WHY notes · 3 = + prompts, vectors, math |
| `/search <query>` | Retrieval only, with no LLM call. Shows the ranking |
| `/chunks` · `/chunk <id>` | List all chunks / read one in full |
| `/term <word>` | What BM25 knows about a word: df, IDF, which chunks contain it |
| `/load <path>` | Ingest another `.txt` / `.md` / `.pdf` from this workspace |
| `/reindex` | Rebuild the index |
| `/history` · `/clear` | Show / forget the conversation |
| `/stats` | Every API call this session, with tokens and estimated cost |
| `/diagram` | Print the flow diagram |
| `/exit` | Quit |

---

## 6. The settings that matter most

All settings live in `rag/config.py`, and each one has a comment explaining it.

| Setting | Default | Effect |
|---|---|---|
| `chunk_size` / `chunk_overlap` | 250 / 40 tokens | Precision vs. context per chunk; robustness at boundaries |
| `chunk_strategy` | `recursive` | `fixed` cuts blindly; `recursive` respects paragraphs and sentences |
| `retrieval_mode` | `hybrid` | `vector` (meaning), `bm25` (keywords), `hybrid` (both) |
| `fusion_method` | `rrf` | `rrf` uses ranks; `weighted` uses `hybrid_alpha` × normalised scores |
| `top_k` | 4 | How many chunks go into the prompt |
| `similarity_threshold` | 0.20 | Drop chunks whose cosine similarity is below this |
| `use_reranker` | false | LLM re-scores the candidates (more precise, extra cost) |
| `use_query_rewriting` | true | Makes follow-up questions searchable |
| `temperature` | 0.2 | Low = factual and stable; high = varied |

---

## 7. Learning path

1. Run the bot at verbosity 3 and ask one question. Read the logs top to bottom alongside `docs/FLOW_DIAGRAM.md`.
2. Read `rag/pipeline.py`, then follow one question through the code.
3. Work through **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**: each lab changes one setting and tells you what to look for.
4. Do the challenge exercises at the end of the labs (multi-query retrieval, persistent index, …).

Every session is also saved in full (verbosity 3) to `logs/session_*.log`, so you can compare two runs side by side.

---

## 8. Notes

* **Cost**: the cost figures in the logs are estimates from the price table in `rag/config.py`. Check OpenAI's pricing page for current numbers. A whole session of questions typically costs well under $0.01 with the default models.
* **Models**: override them in `.env` with `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL`. If you change the embedding model, the vectors change too, and everything is re-embedded.
* **Safety**: only files inside this project folder can be ingested. Never commit your `.env`.
* **Why no vector database or LangChain?** Every piece is written by hand in plain Python and NumPy, so nothing is hidden. Production systems swap in FAISS/pgvector/Qdrant for the vector store and Elasticsearch/OpenSearch for BM25, but the ideas are exactly the same.
