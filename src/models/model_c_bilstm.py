from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .base import BaseRetriever
from .word_vectors import load_word_vectors, sequence_vectors
from ..utils import write_json


class RelevanceNetwork(nn.Module):
    def __init__(self, input_size=300, hidden_size=128, layers=1, dropout=0.3):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True,
                               bidirectional=True, dropout=dropout if layers > 1 else 0.)
        self.classifier = nn.Sequential(nn.Linear(8 * hidden_size, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 1))

    def encode(self, sequences):
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.int64)
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def pair_logits(self, q, d):
        return self.classifier(torch.cat([q, d, torch.abs(q - d), q * d], dim=1)).squeeze(-1)

    def forward(self, queries, documents):
        return self.pair_logits(self.encode(queries), self.encode(documents))


class BiLSTMRetriever(BaseRetriever):
    model_id = "model_c"
    default_threshold = 0.5

    def __init__(self, courses, vectors=None, vector_path=None, **kwargs):
        super().__init__(courses, **kwargs)
        self.vector_path = str(vector_path) if vector_path else None
        self.vectors = vectors if vectors is not None else load_word_vectors(self.vector_path)

    def tensor(self, text):
        array, _, _ = sequence_vectors(text, self.vectors, self.config.get("max_length", 256))
        return torch.from_numpy(array)

    def cache_documents(self):
        self.network.eval()
        with torch.no_grad():
            chunks = []
            for start in range(0, len(self.courses), 16):
                chunks.append(self.network.encode([self.tensor(c["search_text"]) for c in self.courses[start:start + 16]]))
            self.document_embeddings = torch.cat(chunks)

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), self.directory / "weights.pt")
        np.save(self.directory / "document_embeddings.npy", self.document_embeddings.numpy())
        write_json(self.directory / "config.json", self.config)
        self.save_manifest(vector_path=self.vector_path, config=self.config)

    def load(self):
        manifest = self.check_manifest()
        if manifest["vector_path"] != self.vector_path:
            raise ValueError("Use the Word2Vec file used during training.")
        self.config = manifest["config"]
        self.network = RelevanceNetwork(**{k: self.config[k] for k in ("input_size", "hidden_size", "layers", "dropout")})
        self.network.load_state_dict(torch.load(self.directory / "weights.pt", map_location="cpu", weights_only=True))
        self.network.eval()
        self.document_embeddings = torch.from_numpy(np.load(self.directory / "document_embeddings.npy"))
        return self

    def score(self, query):
        array, covered, oov = sequence_vectors(query, self.vectors, self.config["max_length"])
        if not covered:
            return np.zeros(len(self.courses)), {"all_oov": True, "covered_tokens": [], "oov_tokens": oov}
        self.network.eval()
        with torch.no_grad():
            q = self.network.encode([torch.from_numpy(array)]).expand(len(self.courses), -1)
            probabilities = torch.sigmoid(self.network.pair_logits(q, self.document_embeddings)).numpy()
        return probabilities, {"covered_tokens": covered, "oov_tokens": oov,
                               "relevance_probability": float(probabilities.max()), "all_oov": False}
