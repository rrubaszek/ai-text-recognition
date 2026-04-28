import spacy
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class SpacyTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="pl_core_news_sm",
        use_lemma=True,
        remove_stopwords=False,
        remove_punct=True,
        remove_numbers=False,
        lowercase=True,
        batch_size=50
    ):
        self.model_name = model_name
        self.use_lemma = use_lemma
        self.remove_stopwords = remove_stopwords
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.batch_size = batch_size

        self.nlp = None

    def fit(self, X, y=None):
        self.nlp = spacy.load(self.model_name, disable=["ner"])
        return self

    def transform(self, X):
        results = []

        for doc in self.nlp.pipe(X, batch_size=self.batch_size):
            tokens = []

            for token in doc:
                if self.remove_punct and token.is_punct:
                    continue
                if self.remove_stopwords and token.is_stop:
                    continue
                if self.remove_numbers and token.like_num:
                    continue
                if token.is_space:
                    continue

                if self.use_lemma:
                    text = token.lemma_
                else:
                    text = token.text

                if self.lowercase:
                    text = text.lower()
                    
                if text.strip() == "":
                    continue

                tokens.append(text)

            results.append(tokens)

        return results


class SafeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X