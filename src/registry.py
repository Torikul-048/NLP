from .utils import ROOT, load_courses, read_json

MODEL_LABELS = {
    "model_a": "Model A — TF-IDF + Cosine",
    "model_b": "Model B — Pretrained Word2Vec",
    "model_c": "Model C — Word2Vec + BiLSTM",
    "model_d": "Model D — Sentence-BERT",
}


def configured_vector_path():
    import os
    explicit = os.environ.get("WORD2VEC_PATH")
    if explicit:
        return explicit
    config = ROOT / "artifacts/vector_source.json"
    if config.exists():
        path = read_json(config).get("path")
        if path:
            return str((ROOT / path).resolve())
    return None


def create_model(model_id, courses=None, load=True, vector_path=None):
    if load and not (ROOT / "artifacts" / model_id / "manifest.json").exists():
        raise FileNotFoundError(f"No saved {model_id} artifact. Build its index or train Model C first.")
    courses = load_courses() if courses is None else courses
    if model_id == "model_a":
        from .models.model_a_tfidf import TfidfRetriever
        model = TfidfRetriever(courses)
    elif model_id == "model_b":
        from .models.model_b_word2vec import Word2VecRetriever
        model = Word2VecRetriever(courses, vector_path=vector_path or configured_vector_path())
    elif model_id == "model_c":
        from .models.model_c_bilstm import BiLSTMRetriever
        model = BiLSTMRetriever(courses, vector_path=vector_path or configured_vector_path())
    elif model_id == "model_d":
        from .models.model_d_sbert import SentenceBERTRetriever
        model = SentenceBERTRetriever(courses)
    else:
        raise ValueError(f"Unknown model: {model_id}")
    return model.load() if load else model
