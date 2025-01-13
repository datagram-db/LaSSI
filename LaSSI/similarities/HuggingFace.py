import numpy as np
from sentence_transformers import SentenceTransformer, util

class HuggingFace:
    def __init__(self, model=None):
        if model is None:
            model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(model)

    def getEmbedding(self, x):
        return self.model.encode(x)

    def string_similarity(self, x: str, y: str) -> float:  # between 0 and 1
        x_vec = self.model.encode(x)
        y_vec = self.model.encode(y)
        strictSim = float(util.pairwise_dot_score(x_vec, y_vec))
        if (strictSim > np.finfo(float).eps): #ReLU
            return strictSim
        return 0.0