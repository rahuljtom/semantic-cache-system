import faiss
import numpy as np

class VectorStore:

    def __init__(self, embeddings):

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)

    def search(self, query_vector, k=5):

        query_vector = np.array(query_vector).astype("float32")

        D, I = self.index.search(query_vector, k)

        return D, I
