# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight semantic search system built on the 20 Newsgroups dataset.

The system demonstrates how semantic embeddings, fuzzy clustering, and a cluster-aware semantic cache can be combined to efficiently answer natural language queries.

---

## System Architecture

Query
→ Embedding Model
→ Cluster Assignment
→ Semantic Cache Lookup
→ Vector Search (FAISS on cache miss)
→ Response

---

## Embedding Model

We use the model:

sentence-transformers/all-MiniLM-L6-v2

This model provides high-quality semantic embeddings while remaining computationally efficient for real-time search.

---

## Vector Database

The system uses **FAISS** for similarity search over document embeddings.

FAISS allows efficient nearest-neighbor retrieval across ~20,000 documents.

---

## Fuzzy Clustering

The system uses **Gaussian Mixture Models (GMM)** instead of KMeans.

GMM produces **probability distributions over clusters** rather than assigning documents to a single cluster.

Example cluster distribution:

Cluster 3 → 0.61  
Cluster 7 → 0.22  
Cluster 12 → 0.09  

This reflects the overlapping semantic structure of the dataset.

---

## Semantic Cache

Traditional caches fail when queries are phrased differently.

This system implements a **semantic cache**.

Process:

1. Query is embedded
2. Dominant cluster is determined
3. Cache is searched only within that cluster
4. Cosine similarity detects semantically similar queries

If similarity exceeds the threshold (0.85), the cached result is reused.

---

## Tunable Parameter

Semantic similarity threshold:

0.85

Lower values increase cache hit rate but risk incorrect reuse.

Higher values reduce incorrect matches but lower cache efficiency.

---

## API Endpoints

POST /query

Example request:

{
  "query": "space exploration missions"
}

Response includes:

- cache_hit
- similarity_score
- dominant_cluster
- cluster_distribution

---

GET /cache/stats

Returns cache statistics.

---

DELETE /cache

Clears the cache.

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Start the API server:

uvicorn app.main:app --reload
