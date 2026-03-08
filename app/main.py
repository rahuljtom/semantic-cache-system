from fastapi import FastAPI
from pydantic import BaseModel

from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.clustering import ClusterService
from app.semantic_cache import SemanticCache

app = FastAPI()

embedder = EmbeddingService()
vector_db = VectorStore(embedder.embeddings)
clusterer = ClusterService(embedder.embeddings)
cache = SemanticCache()

docs = embedder.docs


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"service": "Semantic Search Cache API running"}


@app.post("/query")
def query(req: QueryRequest):

    q = req.query

    vec = embedder.encode(q)

    cluster_probs = clusterer.distribution(vec)[0]

    top_clusters = sorted(
        [(i, float(p)) for i, p in enumerate(cluster_probs)],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    cluster = clusterer.dominant_cluster(vec)

    cluster_cache_size = len(cache.cache.get(cluster, []))

    cached = cache.lookup(vec, cluster)

    if cached:

        entry, sim = cached

        return {
            "query": q,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": sim,
            "result": entry["result"],
            "dominant_cluster": cluster,
            "cluster_distribution": top_clusters,
            "cache_threshold": cache.threshold,
            "cluster_cache_size": cluster_cache_size
        }

    D, I = vector_db.search(vec)

    results = [docs[i] for i in I[0]]

    cache.add(cluster, q, vec, results)

    return {
        "query": q,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": cluster,
        "cluster_distribution": top_clusters,
        "cache_threshold": cache.threshold,
        "cluster_cache_size": cluster_cache_size
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"status": "cache cleared"}
