from functools import lru_cache
from pathlib import Path

import numpy as np

from ..preprocessing import semantic_tokens
from ..utils import ROOT


@lru_cache(maxsize=2)
def load_word_vectors(path=None):
    from gensim.models import KeyedVectors
    if path:
        path = Path(path)
        if path.suffix == ".kv":
            return KeyedVectors.load(str(path), mmap="r")
        return KeyedVectors.load_word2vec_format(str(path), binary=".bin" in path.name)
    cached = ROOT / ".cache/gensim/word2vec-google-news-300/word2vec-google-news-300.gz"
    if cached.exists():
        return KeyedVectors.load_word2vec_format(str(cached), binary=True)
    import os
    if os.environ.get("ACADEMIC_IR_OFFLINE") == "1":
        raise FileNotFoundError("Word2Vec assets are missing. Run python -m scripts.prepare_assets first.")
    import gensim.downloader as api
    return api.load("word2vec-google-news-300")


def vector_key(token, vectors):
    return next((key for key in (token, token.upper(), token.capitalize()) if key in vectors), None)


def sequence_vectors(text, vectors, max_length=256):
    tokens = semantic_tokens(text)[:max_length]
    keys = [vector_key(t, vectors) for t in tokens]
    # Compact away zero/OOV positions before packing, preserving order of covered words.
    # Thus OOV and padding cannot alter the backward LSTM or pooling.
    known = [vectors[k] for k in keys if k is not None]
    array = np.asarray(known, dtype=np.float32) if known else np.zeros((1, vectors.vector_size), dtype=np.float32)
    return array, [t for t, k in zip(tokens, keys) if k is not None], [t for t, k in zip(tokens, keys) if k is None]
