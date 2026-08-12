# Advanced RAG System

Hybrid retrieval (BM25 + dense vectors, fused with Reciprocal Rank Fusion) with
optional cross-encoder reranking, exposed as a FastAPI service.

The heavy ML stack is optional. Without `sentence-transformers` and `faiss` the
system still builds an index, serves queries and passes its full test suite —
it just falls back to a hashing embedder and an exact numpy index.

## Quick start

```bash
make install          # core deps + dev tools (no torch)
make install-ml       # add real embedding/reranking models (~2GB)

make index            # build the index from data/raw_docs
make serve            # API at http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Why combine BM25 with dense retrieval?", "top_k": 3}'
```

```json
{
  "query": "Why combine BM25 with dense retrieval?",
  "results": [
    {
      "id": "b768f783fd9c3fe5",
      "text": "Dense retrieval embeds queries and documents into the same vector space...",
      "source": "retrieval.txt",
      "position": 1,
      "score": 0.0328,
      "components": {"bm25_rank": 1.0, "dense_rank": 1.0}
    }
  ],
  "elapsed_ms": 3.1,
  "reranked": true
}
```

## How it works

```
documents ──▶ chunker ──▶ embeddings ──┬──▶ dense index (FAISS / numpy) ──┐
                                       │                                  ├──▶ RRF fusion ──▶ cross-encoder rerank ──▶ results
                                       └──▶ BM25 sparse index ────────────┘
```

1. **Chunking** splits documents on progressively finer separators (paragraph →
   line → sentence → word), with word-boundary-aligned overlap so a fact split
   across a boundary stays retrievable.
2. **Dense retrieval** embeds chunks into L2-normalised vectors and searches by
   inner product, i.e. exact cosine similarity.
3. **Sparse retrieval** runs BM25 over lowercased, punctuation-stripped,
   stopword-filtered tokens.
4. **Fusion** merges the two ranked lists with Reciprocal Rank Fusion. RRF
   combines *ranks*, not scores — BM25 scores are unbounded and
   corpus-dependent while cosine similarities live in `[-1, 1]`, so adding them
   directly is meaningless.
5. **Reranking** rescores an over-fetched candidate pool with a cross-encoder
   that reads the query and passage together.

## Configuration

Every setting has a `RAG_*` environment variable override:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_DIR` | `./data` | Root for corpus and index artefacts |
| `RAG_CHUNK_SIZE` | `512` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `RAG_TOP_K` | `5` | Results returned |
| `RAG_CANDIDATE_K` | `30` | Candidates fetched before reranking |
| `RAG_RRF_K` | `60` | RRF damping constant |
| `RAG_DENSE_WEIGHT` / `RAG_SPARSE_WEIGHT` | `1.0` | Fusion weights |
| `RAG_USE_RERANKER` | `true` | Disable to serve fused retrieval order |

Paths are anchored to the repository root, so scripts behave identically
regardless of the directory they are launched from.

## Evaluation

```bash
make evaluate    # or: python scripts/evaluate.py --compare
```

Reports recall@k, precision@k, MRR, nDCG and hit rate for the hybrid pipeline
against BM25-only and dense-only baselines. On the bundled sample corpus
(16 chunks, 8 labelled queries) with `all-MiniLM-L6-v2`:

| Run | recall@5 | recall@10 | nDCG@5 | nDCG@10 | P@3 |
| --- | --- | --- | --- | --- | --- |
| bm25 only | 0.594 | 0.625 | 0.662 | 0.679 | 0.625 |
| dense only | 0.719 | **0.938** | 0.774 | **0.881** | 0.750 |
| hybrid, no rerank | 0.688 | 0.906 | 0.743 | 0.854 | 0.708 |
| hybrid + cross-encoder | **0.813** | 0.906 | **0.839** | 0.887 | **0.833** |

**Read this table with suspicion.** Two honest caveats:

- **Dense-only edges out hybrid** here. That is expected at this scale, not a
  bug: with 16 chunks and `candidate_k=30`, both retrievers return the entire
  corpus, so RRF just averages two complete rankings and BM25's weaker ordering
  dilutes the dense one. Hybrid earns its keep on corpora large enough for the
  two retrievers to disagree about *which* documents are candidates, and on
  queries containing rare identifiers that embeddings miss. Every query in this
  sample set is a fluent natural-language question — the case dense retrieval
  is best at.
- **8 queries is far too few** to separate these numbers from noise.

The one effect that is unambiguous is reranking: it lifts nDCG@5 from 0.743 to
0.839 and P@3 from 0.708 to 0.833 by reordering a fixed candidate pool. If you
want a defensible retrieval comparison, label a few hundred queries over a real
corpus and tune `RAG_SPARSE_WEIGHT` / `RAG_DENSE_WEIGHT` against them.

Label your own queries in `data/eval/qrels.jsonl`:

```json
{"query": "What is self-attention?", "relevant_sources": ["transformers.txt"]}
{"query": "How does RRF work?", "relevant_ids": ["b768f783fd9c3fe5"]}
```

## Project layout

```
src/rag/
  config.py        Settings, repo-root-anchored paths, RAG_* env overrides
  types.py         Chunk / ScoredChunk with provenance
  chunker.py       Recursive character splitter (no langchain dependency)
  embeddings.py    SentenceTransformer + HashingEmbedder fallback
  vector_store.py  FAISS IndexFlatIP + numpy fallback, both cosine
  corpus.py        JSONL chunk store, kept in index order
  retriever.py     BM25 + dense retrieval fused with RRF
  reranker.py      CrossEncoder reranking, no-op fallback
  pipeline.py      Retrieve → rerank orchestration
  evaluator.py     Retrieval and generation metrics
  app.py           FastAPI service
scripts/
  build_index.py     Chunk and index the corpus
  evaluate.py        Measure retrieval quality vs baselines
  optimise_faiss.py  Compress to IVF-PQ and measure the recall cost
  train_reranker.py  Fine-tune the cross-encoder
```

## Scaling up

The default index is exact (`IndexFlatIP`), which is the right choice up to
roughly a million vectors. Beyond that, compress it:

```bash
python scripts/optimise_faiss.py --nprobe 16
```

This trains IVF-PQ on the actual corpus and reports the resulting size
reduction alongside the measured recall loss against the exact index, so the
trade-off is a decision rather than a guess. It refuses to run on corpora too
small to train a meaningful codebook.

## Fine-tuning the reranker

```bash
pip install -e '.[train]'
python scripts/train_reranker.py --data data/training/reranker_data.jsonl
export RAG_RERANKER_MODEL=models/reranker
```

Training data is JSONL of `{"query": ..., "document": ..., "label": 0|1}`.

## Development

```bash
make test     # 78 tests, offline, ~0.3s
make lint
```

The test suite uses a deterministic hashing embedder, so it downloads no models
and needs no network. CI runs on Python 3.10, 3.12 and 3.13.

## License

MIT
