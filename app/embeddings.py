from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import re


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        data = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes")
        )

        docs = []
        for d in data.data:

            d = re.sub(r"http\S+", "", d)
            d = re.sub(r"\S+@\S+", "", d)
            d = re.sub(r"\s+", " ", d)

            docs.append(d.strip())

        self.docs = docs

        self.embeddings = self.model.encode(
            docs,
            show_progress_bar=True
        )


    def encode(self, text):

        return self.model.encode([text])
