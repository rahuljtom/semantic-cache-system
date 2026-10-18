# Semantic Cache System

A research prototype for semantic retrieval and cache reuse over the 20 Newsgroups dataset. It combines sentence embeddings, probabilistic clustering, FAISS nearest-neighbor search, and a similarity threshold to explore when a previously cached result can serve a differently worded query.

**Stack:** Python · sentence-transformers · scikit-learn · FAISS

## Query flow

```text
Query
  → embedding
  → Gaussian mixture cluster assignment
  → cluster-scoped cache lookup
  → FAISS retrieval on cache miss
  → response
```

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`. Gaussian Mixture Models produce a distribution over clusters; the dominant cluster scopes the cache lookup. A cosine-similarity threshold (0.85 in the current prototype) controls whether a cached result is reused.

## API

- `POST /query` — accept a natural-language query; response includes cache status, similarity score, and cluster information
- `GET /cache/stats` — inspect cache statistics
- `DELETE /cache` — clear the cache

Example:

```json
{"query":"space exploration missions"}
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Scope

This is a small experimental project, not a production vector database or a general-purpose semantic cache. The threshold is tunable: lowering it can increase reuse while increasing the risk of a false match. The idea is also being explored as a cache layer for the [LLM Inference Gateway](https://github.com/rahuljtom/llm-inference-gateway).
