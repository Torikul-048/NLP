import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseRetriever
from ..preprocessing import lexical_tokens


class TfidfRetriever(BaseRetriever):
    model_id = "model_a"
    default_threshold = 0.12

    def build(self):
        self.vectorizer = TfidfVectorizer(tokenizer=lexical_tokens, token_pattern=None,
                                         ngram_range=(1, 2), sublinear_tf=True, min_df=1, norm="l2")
        self.matrix = self.vectorizer.fit_transform([c["search_text"] for c in self.courses])
        self.directory.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.vectorizer, self.matrix), self.directory / "index.joblib")
        self.save_manifest()
        return self

    def load(self):
        self.check_manifest()
        self.vectorizer, self.matrix = joblib.load(self.directory / "index.joblib")
        return self

    def score(self, query):
        vector = self.vectorizer.transform([query])
        terms = self.vectorizer.get_feature_names_out()
        important = sorted(zip(vector.indices, vector.data), key=lambda v: -v[1])[:12]
        return (self.matrix @ vector.T).toarray().ravel(), {
            "preprocessed_query": lexical_tokens(query),
            "important_tfidf_terms": {terms[i]: float(w) for i, w in important}}
