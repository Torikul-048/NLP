from __future__ import annotations

import hashlib
import re
import numpy as np

from ..answer_formatter import excerpt, format_answer
from ..query_parser import QueryParser
from ..utils import ROOT, corpus_hash, read_json, write_json


class BaseRetriever:
    model_id = "base"
    default_threshold = 1.0

    def __init__(self, courses, threshold=None):
        self.courses = courses
        self.parser = QueryParser(courses)
        self.fingerprint = corpus_hash(courses)
        self.directory = ROOT / "artifacts" / self.model_id
        self.threshold = self.default_threshold if threshold is None else threshold
        self.calibrated = False
        path = ROOT / "artifacts/thresholds.json"
        if path.exists() and threshold is None:
            data = read_json(path).get(self.model_id, {})
            manifest_path = self.directory / "manifest.json"
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
            if data.get("corpus_hash") == self.fingerprint and data.get("manifest_sha256") == manifest_hash:
                self.threshold = data["threshold"]
                self.calibrated = True

    def save_manifest(self, **kwargs):
        import uuid
        write_json(self.directory / "manifest.json", {"corpus_hash": self.fingerprint,
                   "build_id": str(uuid.uuid4()), "course_codes": [c["course_code"] for c in self.courses], **kwargs})

    def check_manifest(self):
        manifest = read_json(self.directory / "manifest.json")
        if manifest["corpus_hash"] != self.fingerprint:
            raise ValueError("The corpus changed. Rebuild indexes, retrain Model C and recalibrate thresholds.")
        return manifest

    def score(self, query):
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 3):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        p = self.parser.parse(query)
        status = {"EMPTY": "empty", "OUT_OF_SCOPE": "out_of_scope", "GIBBERISH": "gibberish",
                  "UNSUPPORTED_FIELD": "unsupported"}.get(p.intent)
        results, details = [], {"parsed_query": p.as_dict(), "threshold": self.threshold,
                               "threshold_calibrated": self.calibrated}
        if status is None:
            explicit = [c for c in self.courses if p.course_code == c["course_code"]] if p.course_code else []
            if not p.course_code and p.course_title:
                explicit = [c for c in self.courses if c["course_title"].lower() == p.course_title.lower()]
            if explicit and (p.year or p.term):
                explicit = [c for c in explicit if (not p.year or c.get("year") == p.year)
                            and (not p.term or c.get("term") == p.term)]
            if p.course_code and not explicit:
                status = "unsupported"
            elif p.intent == "YEAR_TERM_QUERY" and (p.year or p.term) and not (p.course_code or p.course_title):
                matches = [c for c in self.courses if (not p.year or c.get("year") == p.year)
                           and (not p.term or c.get("term") == p.term)]
                if re.search(r"\btheor(?:y|ies)\b", p.original, re.I):
                    matches = [c for c in matches if c["course_type"] == "Theory"]
                elif re.search(r"\b(project|thesis)\b", p.original, re.I):
                    matches = [c for c in matches if c["course_type"] == "Project"]
                results = self._results([(c, None) for c in matches], p)
                status = "ok" if results else "unsupported"
                details["route"] = "structured_metadata"
            elif p.intent == "LAB_QUERY" and (p.year or p.term) and not (p.course_code or p.course_title):
                matches = [c for c in self.courses if c["course_type"] == "Laboratory"
                           and (not p.year or c.get("year") == p.year)
                           and (not p.term or c.get("term") == p.term)]
                results = self._results([(c, None) for c in matches], p)
                status = "ok" if results else "unsupported"
                details["route"] = "structured_metadata"
            elif explicit:
                if p.intent == "LAB_QUERY":
                    codes = {c["course_code"] for c in explicit}
                    explicit = [c for c in self.courses if c.get("parent_course") in codes or
                                (c["course_code"] in codes and c["course_type"] == "Laboratory")]
                results = self._results([(c, None) for c in explicit], p)
                status = "ok" if results else "unsupported"
                details["route"] = "structured_metadata"
            else:
                scores, debug = self.score(p.search_query)
                details.update(debug)
                details["route"] = "model_retrieval"
                candidates = list(range(len(self.courses)))
                if p.intent == "LAB_QUERY":
                    candidates = [i for i in candidates if self.courses[i]["course_type"] == "Laboratory"]
                elif p.intent not in {"COURSE_CODE_QUERY", "COURSE_DETAILS"}:
                    candidates = [i for i in candidates if not self.courses[i].get("generic_lab")]
                order = sorted(candidates, key=lambda i: (-float(scores[i]), self.courses[i]["course_code"]))
                details["top_score"] = float(scores[order[0]]) if order else None
                if debug.get("all_oov"):
                    status = "insufficient_representation"
                elif not order or scores[order[0]] < self.threshold:
                    status = "no_match"
                else:
                    picked = [(self.courses[i], float(scores[i])) for i in order[:top_k] if scores[i] >= self.threshold]
                    results = self._results(picked, p)
                    status = "ok"
        return {"model": self.model_id, "intent": p.intent, "status": status,
                "answer": format_answer(p, results, status), "results": results, "processing_details": details}

    def _results(self, pairs, parsed):
        return [{**c, "score": score, "matched_excerpt": excerpt(c, parsed.search_query)} for c, score in pairs]
