import numpy as np
from collections import Counter
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from ts2vg import NaturalVG
from tqdm import tqdm


class NonlinearDynamics(BaseEstimator, TransformerMixin):
    """
    Ekstrakcja cech dynamiki nieliniowej (Pillar 3):
    - Taylor's Law (Fluctuation Scaling)
    - RQA na Przypadkach Gramatycznych
    - CRQA / JRQA (Joint Recurrence): Przypadki + Długości Słów
    - NVG na głębokościach składniowych
    - NOWOŚĆ: MSE (Multiscale Entropy Variance) - Wariancja tempa pisania
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for doc in tqdm(X, desc="4/4: Dynamika Nieliniowa (RQA/CRQA/NVG/MSE)"):
            tokens, pos_tags, cases, depths = doc["tokens"], doc["pos"], doc["cases"], doc["depths"]

            if len(tokens) < 10:
                features.append([0.0] * 14)  # Zwiększyliśmy wektor do 14 cech
                continue

            # ---------------------------------------------------------
            # 1. Fluctuation Scaling (Taylor's Law)
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # 2 & 3. RQA (Przypadki) i CRQA (Przypadki + Długości Słów)
            # ---------------------------------------------------------
            seq_cases = cases[:300]
            seq_lens = [len(t) for t in tokens[:300]]
            N_cases = len(seq_cases)

            if N_cases > 5:
                R_cases = np.equal.outer(seq_cases, seq_cases)
                R_lens = np.equal.outer(seq_lens, seq_lens)
                JR = R_cases & R_lens

                np.fill_diagonal(R_cases, False)
                np.fill_diagonal(JR, False)

                tot_rqa = np.sum(R_cases)
                rqa_rr = tot_rqa / (N_cases * N_cases)
                diag_len_rqa, vert_len_rqa = [], []
                for offset in range(1, N_cases):
                    diag = np.diag(R_cases, k=offset)
                    runs = np.diff(np.concatenate(([0], diag.astype(int), [0])))
                    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
                    diag_len_rqa.extend((ends - starts)[(ends - starts) >= 2])
                rqa_det = (sum(diag_len_rqa) * 2) / tot_rqa if tot_rqa > 0 else 0.0

                if diag_len_rqa:
                    p = np.array(list(Counter(diag_len_rqa).values())) / len(diag_len_rqa)
                    rqa_entr = -np.sum(p * np.log(p + 1e-12))
                else:
                    rqa_entr = 0.0

                for col in range(N_cases):
                    column = R_cases[:, col]
                    runs = np.diff(np.concatenate(([0], column.astype(int), [0])))
                    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
                    vert_len_rqa.extend((ends - starts)[(ends - starts) >= 2])
                rqa_lam = sum(vert_len_rqa) / tot_rqa if tot_rqa > 0 else 0.0

                tot_crqa = np.sum(JR)
                crqa_rr = tot_crqa / (N_cases * N_cases)
                diag_len_crqa, vert_len_crqa = [], []
                for offset in range(1, N_cases):
                    diag = np.diag(JR, k=offset)
                    runs = np.diff(np.concatenate(([0], diag.astype(int), [0])))
                    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
                    diag_len_crqa.extend((ends - starts)[(ends - starts) >= 2])
                crqa_det = (sum(diag_len_crqa) * 2) / tot_crqa if tot_crqa > 0 else 0.0

                if diag_len_crqa:
                    p = np.array(list(Counter(diag_len_crqa).values())) / len(diag_len_crqa)
                    crqa_entr = -np.sum(p * np.log(p + 1e-12))
                else:
                    crqa_entr = 0.0

                for col in range(N_cases):
                    column = JR[:, col]
                    runs = np.diff(np.concatenate(([0], column.astype(int), [0])))
                    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
                    vert_len_crqa.extend((ends - starts)[(ends - starts) >= 2])
                crqa_lam = sum(vert_len_crqa) / tot_crqa if tot_crqa > 0 else 0.0

            else:
                rqa_rr, rqa_det, rqa_entr, rqa_lam = 0.0, 0.0, 0.0, 0.0
                crqa_rr, crqa_det, crqa_entr, crqa_lam = 0.0, 0.0, 0.0, 0.0

            # ---------------------------------------------------------
            # 4. NVG (Topologia drzew zależności składniowych)
            # ---------------------------------------------------------
            seq_depths = depths[:300]
            N_dep = len(seq_depths)
            if N_dep > 5:
                try:
                    Y = np.array(seq_depths, dtype=np.float64)
                    Y += np.random.uniform(0, 1e-5, size=Y.shape)

                    vg = NaturalVG()
                    vg.build(Y)
                    G = nx.Graph()
                    G.add_nodes_from(range(N_dep))
                    G.add_edges_from(vg.edges)

                    degrees = [d for n, d in G.degree()]
                    nvg_avg_degree = np.mean(degrees) if degrees else 0.0
                    deg_probs = np.array(list(Counter(degrees).values())) / len(degrees)
                    nvg_graph_entropy = -np.sum(deg_probs * np.log(deg_probs + 1e-12))
                    nvg_clustering = nx.average_clustering(G)

                    components = sorted(nx.connected_components(G), key=len, reverse=True)
                    if len(components) > 0 and len(components[0]) > 1:
                        nvg_avg_path_length = nx.average_shortest_path_length(G.subgraph(components[0]))
                    else:
                        nvg_avg_path_length = 0.0
                except Exception:
                    nvg_avg_degree, nvg_clustering, nvg_avg_path_length, nvg_graph_entropy = 0.0, 0.0, 0.0, 0.0
            else:
                nvg_avg_degree, nvg_clustering, nvg_avg_path_length, nvg_graph_entropy = 0.0, 0.0, 0.0, 0.0

            # ---------------------------------------------------------
            # 5. NOWOŚĆ: MSE (Wariancja Entropii Okienkowej)
            # ---------------------------------------------------------
            window_size = 20
            window_entropies = []

            # Przesuwamy się po tekście skokami co 'window_size'
            for i in range(0, len(tokens), window_size):
                chunk = tokens[i:i + window_size]
                if len(chunk) < 5:  # Pomiń resztki na samym końcu zdania
                    continue

                c_counts = Counter(chunk)
                p = np.array(list(c_counts.values())) / len(chunk)

                # Obliczamy lokalną entropię Shannona dla małego okna
                local_entropy = -np.sum(p * np.log(p + 1e-12))
                window_entropies.append(local_entropy)

            # WARIANCJA: Im wyższa, tym bardziej "ludzki" i zmienny jest tekst
            mse_variance = np.var(window_entropies) if len(window_entropies) > 1 else 0.0

            features.append([
                taylor_b,
                rqa_rr, rqa_det, rqa_entr, rqa_lam,
                crqa_rr, crqa_det, crqa_entr, crqa_lam,
                nvg_avg_degree, nvg_clustering, nvg_avg_path_length, nvg_graph_entropy,
                mse_variance  # <-- Nowa, 14. cecha z tego modułu
            ])

        return np.array(features)