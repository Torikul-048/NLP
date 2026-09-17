from streamlit.testing.v1 import AppTest

from src.utils import ROOT
import pytest


def test_search_and_metadata_rerun():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.selectbox[0].options == ["Model A — TF-IDF + Cosine", "Model B — Pretrained Word2Vec",
                                        "Model C — Word2Vec + BiLSTM", "Model D — Sentence-BERT"]
    app.button(key="search_button").click().run()
    assert not app.exception
    assert "Computer Security" in app.success[0].value
    app.text_input(key="query").set_value("How many credits does NLP have?").run()
    assert "3.0 credits" in app.success[0].value
    app.text_input(key="query").set_value("Who teaches CSE 4121?").run()
    assert not app.exception
    assert "does not provide" in app.info[0].value


def test_same_query_switches_actual_model():
    if not all((ROOT / "artifacts" / f"model_{m}" / "manifest.json").exists() for m in "abcd"):
        pytest.skip("All real model artifacts are required for model-switch integration")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    app.button(key="search_button").click().run()
    initial_query = app.text_input(key="query").value
    for label, model_id in zip(app.selectbox[0].options, ["model_a", "model_b", "model_c", "model_d"]):
        app.selectbox[0].select(label).run()
        assert not app.exception
        assert not app.error
        response = app.session_state["last_response"]
        assert response["model"] == model_id
        assert response["processing_details"]["route"] == "model_retrieval"
        assert app.text_input(key="query").value == initial_query


def test_model_d_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_MODEL_D", "1")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.selectbox[0].select("Model D — Sentence-BERT").run()
    app.button(key="search_button").click().run()
    assert not app.exception
    assert "disabled" in app.warning[0].value
    app.selectbox[0].select("Model A — TF-IDF + Cosine").run()
    assert "Computer Security" in app.success[0].value
