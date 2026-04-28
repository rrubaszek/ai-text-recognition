import numpy as np

from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: Tutaj musimy jeszcze rozszerzyć, bo na razie jest implementacja dla PoC

class NonlinearDynamics(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for tokens in X:
            if len(tokens) < 5:
                features.append([0, 0, 0])
                continue

            counts = Counter(tokens)
            freqs = np.array(list(counts.values()))

            # Burstiness
            burstiness = np.var(freqs) / (np.mean(freqs) + 1e-8)

            # Transition entropy
            transitions = Counter(zip(tokens, tokens[1:]))
            probs = np.array(list(transitions.values()))
            probs = probs / probs.sum()

            trans_entropy = -np.sum(probs * np.log(probs + 1e-12))

            # Recurrence rate
            unique_ratio = len(counts) / len(tokens)
            recurrence = 1 - unique_ratio

            features.append([burstiness, trans_entropy, recurrence])

        return np.array(features)