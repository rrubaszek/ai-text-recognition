import spacy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

class SpacyTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="pl_core_news_sm", use_lemma=True, remove_stopwords=False, remove_punct=True, remove_numbers=False, lowercase=True, batch_size=100):
        self.model_name = model_name
        self.use_lemma = use_lemma
        self.remove_stopwords = remove_stopwords
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.batch_size = batch_size
        self.nlp = None

    def fit(self, X, y=None):
        self.nlp = spacy.load(self.model_name, disable=["ner", "textcat"])
        return self

    def transform(self, X):
        results = []
        # Było: for doc in self.nlp.pipe(X, batch_size=self.batch_size):
        for doc in tqdm(self.nlp.pipe(X, batch_size=self.batch_size), total=len(X), desc="1/4: Tokenizacja (spaCy)"):
            tokens, pos_tags, dep_depths = [], [], []

            for token in doc:
                if self.remove_punct and token.is_punct: continue
                if self.remove_stopwords and token.is_stop: continue
                if self.remove_numbers and token.like_num: continue
                if token.is_space: continue

                # Bezpieczne sprawdzanie głębokości (ochrona przed cyklami w grafie spaCy)
                depth = 0
                curr = token
                visited = set()
                while curr.head != curr and curr not in visited:
                    visited.add(curr)
                    depth += 1
                    curr = curr.head

                text = token.lemma_ if self.use_lemma else token.text
                if self.lowercase: text = text.lower()
                if text.strip() == "": continue

                tokens.append(text)
                pos_tags.append(token.pos_)
                dep_depths.append(depth)

            # wyciąganie długości poszczególnych zdań z parsera spaCy
            sent_lengths = [len(sent.text) for sent in doc.sents]

            results.append({
                "tokens": tokens,
                "pos": pos_tags,
                "depths": dep_depths,
                "raw_char_len": len(doc.text),
                "comma_count": doc.text.count(','),
                "dot_count": doc.text.count('.'),
                "sent_lengths": sent_lengths,  # Cechy zdań
                "raw_text": doc.text
            })
        return results


class SafeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Zabezpieczenie przed błędami
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X