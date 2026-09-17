import numpy as np

from src.models.model_d_sbert import SentenceBERTRetriever
from src.utils import load_courses


def test_sbert_real_index_shape_and_normalization():
    from src.utils import ROOT, read_json
    path = ROOT / "artifacts/model_d/document_embeddings.npy"
    if not path.exists():
        import pytest
        pytest.skip("Run scripts.build_all_indexes --models d for the real embedding check")
    matrix = np.load(path)
    assert matrix.shape == (len(load_courses()), 384)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-5)
    manifest = read_json(path.parent / "manifest.json")
    assert manifest["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
