from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("GENSIM_DATA_DIR", str(ROOT / ".cache" / "gensim"))
os.environ.setdefault("NLTK_DATA", str(ROOT / ".cache" / "nltk"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_courses():
    path = ROOT / "data/processed/courses.json"
    if not path.exists():
        raise FileNotFoundError("Build the corpus first: python -m scripts.build_corpus")
    return read_json(path)


def corpus_hash(courses):
    return hashlib.sha256(json.dumps(courses, sort_keys=True).encode()).hexdigest()


def normalize_rows(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
