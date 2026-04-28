import numpy as np

from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: Tutaj musimy jeszcze rozszerzyć, bo na razie jest implementacja dla PoC

class ShallowStylometrics(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for tokens in X:
            if len(tokens) == 0:
                features.append([0, 0, 0])
                continue

            counts = Counter(tokens)

            ttr = len(counts) / len(tokens)

            hapax = sum(1 for c in counts.values() if c == 1)
            hlr = hapax / len(tokens)

            probs = np.array(list(counts.values())) / len(tokens)
            entropy = -np.sum(probs * np.log(probs + 1e-12))

            features.append([ttr, hlr, entropy])

        return np.array(features)

