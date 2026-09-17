"""Shared deterministic preprocessing; never changes a course-code number."""
from __future__ import annotations

import re
from functools import lru_cache
from difflib import get_close_matches

CODE_RE = re.compile(r"\b([A-Za-z]{2,5})[\s-]*(\d{4})\b")
TOKEN_RE = re.compile(r"[a-z]+(?:[+#]+)?|\d+(?:\.\d+)?", re.I)
TECHNICAL = {"os", "rsa", "pos", "bfs", "dfs", "c", "c++", "c#", "sql", "nlp", "ml"}


def normalize_codes(text):
    return CODE_RE.sub(lambda m: f"{m[1].upper()} {m[2]}", text)


def semantic_tokens(text):
    return TOKEN_RE.findall(normalize_codes(text).lower())


@lru_cache(maxsize=20000)
def lemma(token):
    if token in TECHNICAL:
        return token
    from nltk.stem import WordNetLemmatizer
    try:
        return WordNetLemmatizer().lemmatize(token)
    except LookupError:
        # Offline noun normalization, deliberately conservative.
        return {"systems": "system", "algorithms": "algorithm", "networks": "network",
                "trees": "tree", "languages": "language", "structures": "structure",
                "credits": "credit", "courses": "course"}.get(token, token)


def lexical_tokens(text, remove_stopwords=True):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [lemma(t) for t in semantic_tokens(text)
            if not remove_stopwords or t not in ENGLISH_STOP_WORDS or t in TECHNICAL]


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = nxt
    return row[-1]


def correct_typos(text, vocabulary):
    corrections = {}
    def replace(match):
        word = match[0].lower()
        if len(word) < 6 or word in vocabulary:
            return match[0]
        candidates = [v for v in get_close_matches(word, vocabulary, n=4, cutoff=0.83)
                      if edit_distance(word, v) == 1]
        if len(candidates) == 1:
            corrections[word] = candidates[0]
            return candidates[0]
        return match[0]
    return re.sub(r"[a-zA-Z]+", replace, text), corrections
