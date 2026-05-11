import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

class ZipfFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        features = []
        # Było: for doc in X:
        for doc in tqdm(X, desc="3/4: Analiza Prawa Zipfa"):
            tokens = doc["tokens"]
            if len(tokens) < 3:
                features.append([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
                continue

            counts = Counter(tokens)
            freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
            ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
            log_r, log_f = np.log(ranks), np.log(freqs)

            if len(log_r) < 2 or np.std(log_f) == 0:
                features.append([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
                continue

            model = LinearRegression()
            model.fit(log_r.reshape(-1, 1), log_f)
            pred = model.predict(log_r.reshape(-1, 1))

            mape = np.mean(np.abs((np.exp(log_f) - np.exp(pred)) / np.maximum(np.exp(log_f), 1)))
            corr = np.corrcoef(log_f, pred)[0, 1]
            if np.isnan(corr): corr = 0.0
            slope = model.coef_[0]
            res_var = np.var(log_f - pred)

            emp_pdf = freqs / np.sum(freqs)
            fit_pdf = np.exp(pred) / np.sum(np.exp(pred))
            ks_stat = np.max(np.abs(np.cumsum(emp_pdf) - np.cumsum(fit_pdf)))

            # Metric R (Proxy Prawa Heapsa)
            half_idx = len(tokens) // 2
            metric_r = len(set(tokens[half_idx:]) - set(tokens[:half_idx])) / (len(set(tokens[half_idx:])) + 1e-8)

            features.append([mape, corr, slope, res_var, ks_stat, metric_r])
        return np.array(features)