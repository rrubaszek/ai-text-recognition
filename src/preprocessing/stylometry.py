import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

# Funkcja pomocnicza do liczenia Measure of Textual Lexical Diversity (MTLD)
def calc_mtld(tokens, ttr_threshold=0.72):
    if not tokens: return 0.0

    def evaluate(seq):
        factors, ttr, token_count = 0, 1.0, 0
        types = set()
        for token in seq:
            token_count += 1
            types.add(token)
            ttr = len(types) / token_count
            if ttr < ttr_threshold:
                factors += 1
                types, ttr, token_count = set(), 1.0, 0
        if token_count > 0:
            factors += (1.0 - ttr) / (1.0 - ttr_threshold)
        return len(seq) / factors if factors > 0 else 0.0

    # MTLD liczy się standardowo od przodu i od tyłu, a wynik uśrednia
    return (evaluate(tokens) + evaluate(tokens[::-1])) / 2.0


class ShallowStylometrics(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        # Było: for doc in X:
        for doc in tqdm(X, desc="2/4: Płytka Stylometria"):
            tokens, pos_tags, depths = doc["tokens"], doc["pos"], doc["depths"]
            N = len(tokens)

            raw_len = doc.get("raw_char_len", 0)
            token_count = N
            comma_ratio = doc.get("comma_count", 0) / (N + 1e-5)
            dot_ratio = doc.get("dot_count", 0) / (N + 1e-5)
            sent_lengths = doc.get("sent_lengths", [])
            avg_sent_len = np.mean(sent_lengths) if len(sent_lengths) > 0 else 0.0
            var_sent_len = np.var(sent_lengths) if len(sent_lengths) > 0 else 0.0

            # Zabezpieczenie dla pustych tekstów
            if N == 0:
                # [6 pierwszych cech] + 12 zer = równe 18 kolumn
                features.append([token_count, raw_len, comma_ratio, dot_ratio, avg_sent_len, var_sent_len] + [0.0] * 12)
                continue

            counts = Counter(tokens)
            ttr = len(counts) / N
            hlr = sum(1 for c in counts.values() if c == 1) / N
            probs = np.array(list(counts.values())) / N
            entropy = -np.sum(probs * np.log(probs + 1e-12))

            word_lengths = np.array([len(t) for t in tokens])
            avg_word_len, var_word_len = np.mean(word_lengths), np.var(word_lengths)

            # Yule's K
            m1, m2 = N, sum(c ** 2 for c in counts.values())
            yules_k = 10000 * (m2 - m1) / (m1 ** 2) if m1 > 1 else 0.0

            # Wywołanie MTLD dla obecnego dokumentu
            mtld = calc_mtld(tokens)

            pos_counts = Counter(pos_tags)
            noun_ratio = pos_counts.get("NOUN", 0) / N
            verb_ratio = pos_counts.get("VERB", 0) / N
            adj_ratio = pos_counts.get("ADJ", 0) / N
            avg_dep_depth, max_dep_depth = (np.mean(depths), np.max(depths)) if len(depths) > 0 else (0.0, 0.0)

            # Dodajemy wszystko do tablicy (kolejność musi pasować do `feature_names` z main.py)
            features.append([
                token_count, raw_len, comma_ratio, dot_ratio, avg_sent_len, var_sent_len,
                ttr, hlr, entropy, avg_word_len, var_word_len,
                yules_k, mtld,  # OBA WSKAŹNIKI OBECNE
                noun_ratio, verb_ratio, adj_ratio, avg_dep_depth, max_dep_depth
            ])

        return np.array(features)