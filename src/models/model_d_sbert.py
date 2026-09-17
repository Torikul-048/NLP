from functools import lru_cache

import numpy as np

from .base import BaseRetriever


@lru_cache(maxsize=2)
def load_encoder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")


class SentenceBERTRetriever(BaseRetriever):
    model_id = "model_d"
    default_threshold = 0.4

    def __init__(self, courses, model_name="sentence-transformers/all-MiniLM-L6-v2", encoder=None, **kwargs):
        super().__init__(courses, **kwargs)
        self.model_name = model_name
        self.encoder = encoder if encoder is not None else load_encoder(model_name)

    def build(self):
        self.matrix = self.encoder.encode([c["search_text"] for c in self.courses],
                                         normalize_embeddings=True, show_progress_bar=True, batch_size=16)
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "document_embeddings.npy", self.matrix)
        self.save_manifest(model_name=self.model_name, dimension=self.matrix.shape[1],
                           max_sequence_length=self.encoder.max_seq_length,
                           truncated_documents=sum(len(self.encoder.tokenizer.encode(c["search_text"])) > self.encoder.max_seq_length for c in self.courses))
        return self

    def load(self):
        manifest = self.check_manifest()
        if manifest["model_name"] != self.model_name:
            raise ValueError("Sentence-BERT encoder differs from the saved index.")
        self.matrix = np.load(self.directory / "document_embeddings.npy")
        return self

    def score(self, query):
        vector = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.matrix @ vector, {"embedding_dimension": len(vector), "similarity": "cosine"}
