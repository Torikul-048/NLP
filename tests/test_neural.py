import numpy as np
import pytest

torch = pytest.importorskip("torch")
gensim = pytest.importorskip("gensim")

from src.models.model_c_bilstm import RelevanceNetwork
from src.models.model_b_word2vec import Word2VecRetriever
from src.models.word_vectors import sequence_vectors


def test_packed_encoder_ignores_other_sequences_padding():
    torch.manual_seed(2)
    model = RelevanceNetwork(input_size=4, hidden_size=3, dropout=0.)
    model.eval()
    short, long = torch.randn(2, 4), torch.randn(9, 4)
    alone = model.encode([short])[0]
    batched = model.encode([short, long])[0]
    assert torch.allclose(alone, batched, atol=1e-6)
    assert model([short], [long]).shape == (1,)


def test_vectors_keep_pretrained_words_outside_corpus(tmp_path):
    from gensim.models import KeyedVectors
    vectors = KeyedVectors(vector_size=3)
    vectors.add_vectors(["encryption", "cryptography", "music"], np.eye(3, dtype=np.float32))
    courses = [{"course_code": "CSE 1001", "course_title": "Cryptography", "description": "cryptography",
                "search_text": "cryptography", "course_type": "Theory"}]
    model = Word2VecRetriever(courses, vectors=vectors)
    model.directory = tmp_path
    model.build()
    query_vector, covered, _ = model.embed("encryption")
    assert covered == ["encryption"] and np.linalg.norm(query_vector) > 0
    assert model.score("zzzzzzzz")[1]["all_oov"]
    sequence, covered, oov = sequence_vectors("encryption zzz music", vectors)
    assert sequence.shape == (2, 3)
    assert oov == ["zzz"]


def test_bilstm_learns_toy_pair_signal():
    # Functional gradient test with synthetic tensors, never used as pretrained retrieval vectors.
    torch.manual_seed(3)
    torch.set_num_threads(2)
    net = RelevanceNetwork(input_size=3, hidden_size=4, dropout=0.)
    q = [torch.ones(2, 3), -torch.ones(2, 3)]
    d = [torch.ones(3, 3), torch.ones(3, 3)]
    labels = torch.tensor([1., 0.])
    optimizer = torch.optim.Adam(net.parameters(), lr=.03)
    criterion = torch.nn.BCEWithLogitsLoss()
    initial = criterion(net(q, d), labels).item()
    for _ in range(35):
        optimizer.zero_grad()
        loss = criterion(net(q, d), labels)
        loss.backward()
        optimizer.step()
    assert criterion(net(q, d), labels).item() < initial * .5
