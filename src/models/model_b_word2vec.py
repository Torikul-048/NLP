import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseRetriever
from .word_vectors import load_word_vectors, vector_key
from ..preprocessing import semantic_tokens
from ..utils import normalize_rows


class Word2VecRetriever(BaseRetriever):
    model_id = "model_b"
    default_threshold = 0.45

    def __init__(self, courses, vectors=None, vector_path=None, **kwargs):
        super().__init__(courses, **kwargs)
        self.vector_path = str(vector_path) if vector_path else None
        self.vectors = vectors if vectors is not None else load_word_vectors(self.vector_path)

    def embed(self, text):
        from collections import Counter
        counts = Counter(semantic_tokens(text))
        total, weights, covered, oov = np.zeros(self.vectors.vector_size, dtype=np.float32), 0., [], []
        for token, frequency in counts.items():
            key = vector_key(token, self.vectors)
            if key is None:
                oov.append(token)
                continue
            # Unseen query words retain pretrained semantics using smoothed unseen-term IDF.
            index = self.vectorizer.vocabulary_.get(token)
            idf = self.vectorizer.idf_[index] if index is not None else self.unseen_idf
            weight = (1 + np.log(frequency)) * idf
            total += weight * self.vectors[key]
            weights += weight
            covered.append(token)
        return normalize_rows(total / max(weights, 1e-12)), covered, oov

    def build(self):
        self.vectorizer = TfidfVectorizer(tokenizer=semantic_tokens, token_pattern=None, sublinear_tf=True)
        self.vectorizer.fit([c["search_text"] for c in self.courses])
        self.unseen_idf = float(np.log(1 + len(self.courses)) + 1)
        self.matrix = np.stack([self.embed(c["search_text"])[0] for c in self.courses])
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "document_vectors.npy", self.matrix)
        joblib.dump(self.vectorizer, self.directory / "vectorizer.joblib")
        self.save_manifest(vector_path=self.vector_path, vector_size=self.vectors.vector_size,
                           unseen_idf=self.unseen_idf, pretrained="word2vec-google-news-300" if not self.vector_path else self.vector_path)
        return self

    def load(self):
        data = self.check_manifest()
        self.unseen_idf = data["unseen_idf"]
        self.vectorizer = joblib.load(self.directory / "vectorizer.joblib")
        self.matrix = np.load(self.directory / "document_vectors.npy")
        if data["vector_path"] != self.vector_path:
            raise ValueError("Use the same Word2Vec file used to build this index.")
        return self

    def score(self, query):
        vector, covered, oov = self.embed(query)
        return self.matrix @ vector, {"covered_tokens": covered, "oov_tokens": oov, "all_oov": not covered}
