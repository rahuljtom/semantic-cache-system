import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.threshold = threshold
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, vector, cluster):

        if cluster not in self.cache:
            self.miss_count += 1
            return None

        entries = self.cache[cluster]

        for item in entries:

            sim = cosine_similarity(vector, item["embedding"])[0][0]

            if sim >= self.threshold:
                self.hit_count += 1
                return item, float(sim)

        self.miss_count += 1
        return None

    def add(self, cluster, query, embedding, result):

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append(
            {
                "query": query,
                "embedding": embedding,
                "result": result
            }
        )

    def stats(self):

        total = self.hit_count + self.miss_count

        if total == 0:
            rate = 0
        else:
            rate = self.hit_count / total

        entries = sum(len(v) for v in self.cache.values())

        return {
            "total_entries": entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": rate
        }

    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
