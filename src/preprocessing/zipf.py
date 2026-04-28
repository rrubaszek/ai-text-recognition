import numpy as np

from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: Tutaj musimy jeszcze rozszerzyć, bo na razie jest implementacja dla PoC

class ZipfFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for tokens in X:
            if len(tokens) < 5:
                features.append([0.0, 0.0])
                continue

            counts = Counter(tokens)
            freqs = np.array(sorted(counts.values(), reverse=True))

            ranks = np.arange(1, len(freqs) + 1)

            log_r = np.log(ranks)
            log_f = np.log(freqs)

            # guard
            if len(log_r) < 2 or np.std(log_f) == 0:
                features.append([0.0, 0.0])
                continue

            model = LinearRegression()
            model.fit(log_r.reshape(-1, 1), log_f)

            pred = model.predict(log_r.reshape(-1, 1))

            # MAPE (stable)
            mape = np.mean(
                np.abs((np.exp(log_f) - np.exp(pred)) / np.maximum(np.exp(log_f), 1))
            )

            # correlation (clean)
            corr = np.corrcoef(log_f, pred)[0, 1]
            if np.isnan(corr):
                corr = 0.0

            features.append([mape, corr])

        return np.array(features)