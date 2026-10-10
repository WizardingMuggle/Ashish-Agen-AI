# RAG Lab: Guided Experiments

Each experiment changes one setting with `/set`, asks a question, and asks you to compare the logs **before and after**.
Write down what you observe; the "What to look for" notes tell you where in the logs to look.

> Tips
> * `/search <query>` runs retrieval **without** calling the LLM. It is free, apart from one embedding call, and fast, so use it a lot.
> * `/config` shows every current value. Restart the bot to get the defaults back.
> * If the logs are too long, scroll back to the stage header you care about (e.g. `[Q6] HYBRID FUSION`).
>   The full log of every session is also saved in `logs/`.

---

## Lab 1: Keywords vs. meaning (BM25 vs. vector search)

| Step | Command |
|---|---|
| 1 | `/set retrieval_mode bm25` |
| 2 | `/search How many vacation days do I get?` |
| 3 | `/set retrieval_mode vector` |
| 4 | `/search How many vacation days do I get?` |

**What to look for:** In `[Q5]`, the query-terms table shows that `vacation` has **df = 0**. That word never appears in the document, which says "annual leave" instead. BM25 can only match on `day`, so its ranking is weak. Vector search in `[Q4]` should rank the "3.1 Annual leave" chunk near the top, because embeddings capture *meaning*.

Now do the reverse:

| Step | Command |
|---|---|
| 5 | `/set retrieval_mode bm25` then `/search E-4471` |
| 6 | `/set retrieval_mode vector` then `/search E-4471` |

**What to look for:** `e-4471` has df = 1, which gives it a very high IDF, so BM25 nails it. Vector search usually finds the error-code section too, but compare the cosine gap between rank #1 and rank #2. Exact codes, IDs and names are where keyword search shines.

Finally, `/set retrieval_mode hybrid` and repeat both searches. Look at the fusion table in `[Q6]`: which chunks are "found by BOTH"?

---

## Lab 2: Chunk size

| Step | Command |
|---|---|
| 1 | `/set chunk_size 60` (re-indexes automatically) |
| 2 | Ask: `What are the specs of the Sparrow drone?` |
| 3 | `/set chunk_size 600` |
| 4 | Ask the same question |

**What to look for:**
* `[I3]`: how many chunks? What is the mean chunk size?
* `[I4]`: cache HIT/MISS. New chunk boundaries mean new texts, so you pay for new embeddings.
* `[Q9]`: the "retrieved context" token count in the prompt budget table.
* With **tiny chunks**, the specs list is spread over many chunks, and `top_k=4` may not collect all of it.
* With **huge chunks**, every chunk mixes several topics, the cosine scores get closer together, and the prompt becomes expensive.

Set it back with `/set chunk_size 250`.

---

## Lab 3: Chunk overlap

| Step | Command |
|---|---|
| 1 | `/set chunk_overlap 0` |
| 2 | `/chunks`, and look at where each chunk starts |
| 3 | `/set chunk_overlap 100` |
| 4 | `/chunks` again |

**What to look for:** In `[I3]`, the "duplication from overlap" percentage changes, and so does the yellow "Overlap between chunk 0 and chunk 1" panel at verbosity 3. More overlap means fewer facts cut in half at a boundary, but more tokens to embed and store.

---

## Lab 4: Fixed vs. recursive chunking

| Step | Command |
|---|---|
| 1 | `/set chunk_strategy fixed` |
| 2 | `/chunk 3` |
| 3 | `/set chunk_strategy recursive` |
| 4 | `/chunk 3` |

**What to look for:** Fixed chunks start and end **mid-sentence, even mid-word**. Recursive chunks end on paragraph or sentence boundaries. Which one would you rather paste into a prompt?

---

## Lab 5: Section titles in chunks ("contextual chunk headers")

| Step | Command |
|---|---|
| 1 | `/search how many days can I take off when I am ill` and note the cosine of the sick-leave chunk |
| 2 | `/set prepend_section_title false` |
| 3 | Repeat the search |

**What to look for:** Without the `Section: 3. Leave Policy > 3.3 Sick leave` prefix, a chunk that begins in the middle of a section (e.g. "A medical certificate is required only when…") no longer says what topic it belongs to. Compare the scores in `[Q4]`, and use `/chunk <id>` to read the chunks involved.

---

## Lab 6: top_k and the similarity threshold

| Step | Command |
|---|---|
| 1 | `/set top_k 1` then ask `What leave types exist besides annual leave?` |
| 2 | `/set top_k 8` then ask again |
| 3 | `/set similarity_threshold 0.5` then ask again |

**What to look for:**
* With `top_k 1` the answer is incomplete, because the information spans several chunks.
* With `top_k 8` the answer is complete, but check `[Q11] Sources`: how many chunks were **not cited**? Those were wasted tokens.
* With a high threshold, `[Q8]` drops chunks. If everything is dropped, the bot says "I couldn't find that in the document."

---

## Lab 7: Fusion: RRF vs. weighted

| Step | Command |
|---|---|
| 1 | `/set fusion_method rrf` then `/search Sparrow wind limit` |
| 2 | `/set fusion_method weighted` then repeat |
| 3 | `/set hybrid_alpha 0.9` then repeat (mostly vector) |
| 4 | `/set hybrid_alpha 0.1` then repeat (mostly BM25) |

**What to look for:** The `FUSED` column in the `[Q6]` table. RRF scores are tiny (≈0.016–0.033) because they come from `1/(60+rank)`. Weighted scores fall between 0 and 1. Does the final order change?

---

## Lab 8: BM25 parameters

| Step | Command |
|---|---|
| 1 | `/term leave`, `/term the`, `/term YubiKey`: compare df and IDF |
| 2 | `/set bm25_remove_stopwords false` then `/search what is the per diem` |
| 3 | `/set bm25_use_stemming false` then `/search policies for drones` |
| 4 | `/set bm25_b 0` (no length penalty) and `/set bm25_k1 0.1` (presence only) |

**What to look for:** At verbosity 3, the "BM25 score breakdown" table in `[Q5]` shows each term's contribution per chunk. Without stopword removal, `the`, `is` and `what` add score noise. Without stemming, `drones` no longer matches `drone`.

---

## Lab 9: Conversational memory and query rewriting

1. Ask: `How long can the Sparrow drone fly?`
2. Ask: `And what's its wind limit?`
3. Look at `[Q2]`: what did the follow-up get rewritten to?
4. `/clear`, then `/set use_query_rewriting false`, and repeat steps 1–2.

**What to look for:** Without rewriting, retrieval searches for `And what's its wind limit?`. BM25 only gets `wind` and `limit`, and the embedding has no idea what "its" means. Check whether the Sparrow chunk is still retrieved.

---

## Lab 10: Hallucination guard

1. Ask: `Who is the CEO of Nimbus Robotics?` (answer in document)
2. Ask: `What is the capital of France?` (not in the document)
3. Ask: `What is Nimbus Robotics' revenue?` (sounds plausible, not in the document)

**What to look for:** In `[Q4]`, the "gap top1 - mean" is small for off-topic questions. The system prompt in `[Q9]` tells the model to answer "I couldn't find that in the document." Then edit `SYSTEM_PROMPT` in `rag/generation/prompt_builder.py`, remove rule 3, restart, and ask question 3 again. What happens?

---

## Lab 11: Reranking

1. `/set use_reranker true`
2. Ask: `What should I do if the robot keeps overheating?`

**What to look for:** The `[Q7]` table's "movement" column shows how the LLM judge re-ordered the fused candidates, and why. Then look at `/stats`: what did the extra rerank call cost?

---

## Lab 12: Temperature

1. `/set temperature 0` and ask `Describe the Atlas robot` twice.
2. `/set temperature 1.2` and ask it twice more.

**What to look for:** At 0 the two answers are (almost) identical. At 1.2 the wording varies. For factual Q&A over documents, keep it low.

---

## Lab 13: Bring your own document

Put any `.txt`, `.md` or `.pdf` file inside the project folder (for example `data/my_notes.pdf`), then:

```
/load data/my_notes.pdf
```

Try a PDF and look at how much `[I2] CLEAN` removes. Then try `/load ../somewhere_else.txt` and see why the bot refuses.

---

## Challenge exercises (code changes)

1. **Add a new cleaning rule** in `rag/ingestion/cleaner.py`, for example removing markdown `**bold**` markers. Does BM25 change?
2. **Add Euclidean search**: in `vector_store.py`, rank by `euclidean_distance` instead of cosine. Is the ranking identical? (It should be. Why?)
3. **Multi-query retrieval**: in `query_rewriter.py`, ask the LLM for 3 different phrasings, retrieve for each, and fuse them all with RRF.
4. **Persist the index**: save the chunk vectors to `.rag_cache/index.npy` and skip re-embedding entirely on start-up.
5. **Add a "page" field** to `Chunk` for PDFs, so answers can cite page numbers.
