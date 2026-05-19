import numpy as np
from collections import Counter
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from ts2vg import NaturalVG
from tqdm import tqdm


class NonlinearDynamics(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        # Było: for doc in X:
        for doc in tqdm(X, desc="4/4: Nieliniowa Dynamika (RQA/NVG)"):
            tokens, pos_tags = doc["tokens"], doc["pos"]
            if len(tokens) < 10:
                features.append([0.0] * 9)
                continue

            # 1. Taylor's Law
            chunk_size = max(10, len(tokens) // 10)
            chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
            means, variances = [], []
            for chunk in chunks:
                freqs = list(Counter(chunk).values())
                if len(freqs) > 1:
                    means.append(np.mean(freqs))
                    variances.append(np.var(freqs))
            means, variances = np.array(means), np.array(variances)
            valid = (means > 0) & (variances > 0)
            if np.sum(valid) >= 3:
                try:
                    taylor_b = \
                    LinearRegression().fit(np.log(means[valid]).reshape(-1, 1), np.log(variances[valid])).coef_[0]
                except Exception:
                    taylor_b = 0.0
            else:
                taylor_b = 0.0

            # 2. RQA (na tagach POS)
            seq_pos = pos_tags[:300]
            N_pos = len(seq_pos)
            if N_pos > 5:
                R = np.equal.outer(seq_pos, seq_pos)
                np.fill_diagonal(R, False)
                total_recurrence = np.sum(R)
                rqa_rr = total_recurrence / (N_pos * N_pos)

                diag_lengths = []
                for offset in range(1, N_pos):
                    diag = np.diag(R, k=offset)
                    runs = np.diff(np.concatenate(([0], diag.astype(int), [0])))
                    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
                    lengths = ends - starts
                    diag_lengths.extend(lengths[lengths >= 2])
                rqa_det = (sum(diag_lengths) * 2) / total_recurrence if total_recurrence > 0 else 0.0

                if len(diag_lengths) > 0:
                    counts = Counter(diag_lengths)
                    probs = np.array(list(counts.values())) / len(diag_lengths)
                    rqa_entr = -np.sum(probs * np.log(probs + 1e-12))
                else:
                    rqa_entr = 0.0

                vert_lengths = []
                for col in range(N_pos):
                    column = R[:, col]
                    runs = np.diff(np.concatenate(([0], column.astype(int), [0])))
                    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
                    lengths = ends - starts
                    vert_lengths.extend(lengths[lengths >= 2])
                rqa_lam = sum(vert_lengths) / total_recurrence if total_recurrence > 0 else 0.0
            else:
                rqa_rr, rqa_det, rqa_entr, rqa_lam = 0.0, 0.0, 0.0, 0.0

            # 3. NVG (Topologia grafu)
            seq_tok = tokens[:300]
            N_tok = len(seq_tok)
            if N_tok > 5:
                try:
                    Y = np.array([len(t) for t in seq_tok], dtype=np.float64)
                    vg = NaturalVG()
                    vg.build(Y)
                    G = nx.Graph()
                    G.add_nodes_from(range(N_tok))
                    G.add_edges_from(vg.edges)

                    degrees = [d for n, d in G.degree()]
                    nvg_avg_degree = np.mean(degrees) if degrees else 0.0

                    deg_counts = Counter(degrees)
                    deg_probs = np.array(list(deg_counts.values())) / len(degrees)
                    nvg_graph_entropy = -np.sum(deg_probs * np.log(deg_probs + 1e-12))

                    nvg_clustering = nx.average_clustering(G)

                    components = sorted(nx.connected_components(G), key=len, reverse=True)
                    if len(components) > 0 and len(components[0]) > 1:
                        G_sub = G.subgraph(components[0])
                        nvg_avg_path_length = nx.average_shortest_path_length(G_sub)
                    else:
                        nvg_avg_path_length = 0.0
                except Exception:
                    nvg_avg_degree, nvg_clustering, nvg_avg_path_length, nvg_graph_entropy = 0.0, 0.0, 0.0, 0.0
            else:
                nvg_avg_degree, nvg_clustering, nvg_avg_path_length, nvg_graph_entropy = 0.0, 0.0, 0.0, 0.0

            features.append(
                [taylor_b, rqa_rr, rqa_det, rqa_entr, rqa_lam, nvg_avg_degree, nvg_clustering, nvg_avg_path_length,
                 nvg_graph_entropy])
        return np.array(features)