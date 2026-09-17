import copy

import numpy as np
import pytest

from src.answer_formatter import NO_INFO
from src.corpus.corpus_builder import parse_pages
from src.datasets import make_pairs, read_queries, validate_datasets
from src.evaluation import calibrate_threshold, retrieval_metrics
from src.models.model_a_tfidf import TfidfRetriever
from src.preprocessing import correct_typos, lexical_tokens, normalize_codes
from src.query_parser import QueryParser
from src.utils import ROOT, load_courses


@pytest.fixture(scope="module")
def courses():
    return load_courses()


@pytest.fixture(scope="module")
def model(courses):
    return TfidfRetriever(courses, threshold=0.12).load()


def test_source_complete(courses):
    assert len(courses) == 108
    assert len({c["course_code"] for c in courses}) == 107
    assert all(c["credits"] is not None and c["contact_hours"] for c in courses)
    assert len({c["record_id"] for c in courses}) == 108
    assert all("Summary of" not in c["description"] and "Credits:" not in c["description"] for c in courses)
    nlp = next(c for c in courses if c["course_code"] == "CSE 4121")
    assert (nlp["source_page"], nlp["year"], nlp["term"], nlp["optional_group"]) == (13, "4th Year", "1st Term", "Optional-I")
    assert next(c for c in courses if c["course_code"] == "CSE 3106")["course_type"] == "Laboratory"


def test_continuation_and_missing_colon():
    courses = parse_pages([
        {"page": 1, "text": "Syllabus of 4thYear 2ndTerm Courses\nCSE 4245 Principles of Programming Languages\nCredits: 3.0\nContact Hours: 3L+0P Hrs/Week\nFunctional languages."},
        {"page": 2, "text": "2\nConcurrent programming.\nSummary of 1st Year 1st Term Courses\nCSE 1101\n3\nSyllabus of 1st Year 1st Term Courses\nCSE 1101: Programming\nCredits: 3.0\nContact Hours: 3L+0P Hrs/Week\nPointers."}])
    assert len(courses) == 2
    assert courses[0]["source_pages"] == [1, 2]
    assert courses[0]["description"] == "Functional languages. Concurrent programming."


@pytest.mark.parametrize("query", ["CSE4121", "CSE-4121", "CSE 4121", "cse4121"])
def test_codes(query):
    assert normalize_codes(query) == "CSE 4121"


def test_technical_terms():
    assert "rsa" in lexical_tokens("RSA and CSE4121")
    assert "4121" in lexical_tokens("CSE4121")
    text, changes = correct_typos("cryptograpy algoritm", {"cryptography", "algorithm"})
    assert text == "cryptography algorithm"
    assert len(changes) == 2
    assert correct_typos("abcdef", {"abcdeg", "abcdeh"})[0] == "abcdef"


def test_parser(courses):
    parser = QueryParser(courses)
    p = parser.parse("which course teaches cryptograpy")
    assert p.search_query == "cryptography"
    assert p.corrections == {"cryptograpy": "cryptography"}
    assert parser.parse("How many credits does NLP have?").search_query == "Natural Language Processing"
    assert parser.parse("Who teaches CSE 4121?").intent == "UNSUPPORTED_FIELD"


@pytest.mark.parametrize("query,status", [
    ("", "empty"), ("  ", "empty"), ("asdf qwer zx", "gibberish"),
    ("Who is the president of France?", "out_of_scope"), ("CSE9999", "unsupported"),
    ("What are the prerequisites for NLP?", "unsupported"),
    ("Which course teaches marine biology?", "no_match")])
def test_rejections(model, query, status):
    result = model.retrieve(query)
    assert result["status"] == status
    assert result["results"] == []


def test_actual_facts_and_lab(model):
    response = model.retrieve("CSE 4121 has 4 credits right?")
    assert "3.0 credits" in response["answer"]
    assert response["results"][0]["score"] is None
    lab = model.retrieve("Does CSE 4115 have a lab?")
    assert [r["course_code"] for r in lab["results"]] == ["CSE 4116"]
    capstone = model.retrieve("How many credits does CSE4000 have?")
    assert "1.5 credits" in capstone["answer"] and "3.0 credits" in capstone["answer"]
    conflict = model.retrieve("What is the credit of HUM4207?")
    assert "Source conflict" in conflict["answer"] and "2.0" in conflict["answer"]


def test_year_term_all_results(model):
    response = model.retrieve("Which courses are in fourth year first term?")
    assert len(response["results"]) == 21
    assert all(c["year"] == "4th Year" and c["term"] == "1st Term" for c in response["results"])
    specific = model.retrieve("Tell me about CSE4000 in second term")
    assert len(specific["results"]) == 1
    assert specific["results"][0]["credits"] == 3.


def test_exact_security(model):
    result = model.retrieve("Which course covers RSA and ElGamal?")
    assert result["results"][0]["course_code"] == "CSE 4115"
    assert "RSA" in result["results"][0]["matched_excerpt"]


def test_metrics_multi_gold():
    m = retrieval_metrics(["A", "B", "C", "D"], {"B", "D"})
    assert m == {"Hit@1": 0., "Hit@3": 1., "MRR": .5, "Recall@3": .5}
    assert retrieval_metrics([], {"B"})["MRR"] == 0.
    assert retrieval_metrics(["A"], set())["MRR"] is None


def test_split_and_negative_integrity(courses, model):
    relevance = read_queries(ROOT / "data/queries/relevance_queries.csv")
    evaluation = read_queries(ROOT / "data/queries/evaluation_queries.csv")
    audit = validate_datasets(relevance, evaluation, courses)
    assert audit["test"] == 48
    pairs = make_pairs([{"query_id": "X", "query_text": "public key encryption", "positive_course_code": "CSE 4115|CSE 4215"}], courses, model)
    assert all(p["course_code"] not in {"CSE 4115", "CSE 4215"} for p in pairs if p["label"] == 0)
    bad = copy.deepcopy(evaluation)
    bad[0]["group_id"] = relevance[0]["group_id"]
    with pytest.raises(ValueError, match="leakage"):
        validate_datasets(relevance, bad, courses)
    with pytest.raises(ValueError, match="validation"):
        calibrate_threshold(model, relevance[:1])


def test_stale_artifact_refused(courses):
    modified = copy.deepcopy(courses)
    modified[0]["credits"] = 99
    with pytest.raises(ValueError, match="corpus changed"):
        TfidfRetriever(modified).load()
