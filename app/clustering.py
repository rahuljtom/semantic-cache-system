from sklearn.mixture import GaussianMixture
import numpy as np

class ClusterService:

    def __init__(self, embeddings):

        self.model = GaussianMixture(
            n_components=30,
            covariance_type="full"
        )

        self.model.fit(embeddings)

    def distribution(self, vector):

        return self.model.predict_proba(vector)

    def dominant_cluster(self, vector):

        probs = self.model.predict_proba(vector)

        return int(np.argmax(probs))
