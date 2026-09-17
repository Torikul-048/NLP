from __future__ import annotations

import csv
import hashlib

import numpy as np

from .datasets import gold_codes
from .utils import ROOT, write_json


def ranked_codes(model, query):
    parsed = model.parser.parse(query)
    scores, debug = model.score(parsed.search_query or query)
    eligible = list(range(len(model.courses)))
    if parsed.intent == "LAB_QUERY":
        eligible = [i for i in eligible if model.courses[i]["course_type"] == "Laboratory"]
    elif parsed.intent not in {"COURSE_CODE_QUERY", "COURSE_DETAILS", "CREDIT_QUERY", "CONTACT_HOUR_QUERY", "YEAR_TERM_QUERY"}:
        eligible = [i for i in eligible if not model.courses[i].get("generic_lab")]
    order = sorted(eligible, key=lambda i: (-float(scores[i]), model.courses[i]["course_code"]))
    unique = list(dict.fromkeys(model.courses[i]["course_code"] for i in order))
    top = float(scores[order[0]]) if order else 0.
    return unique, top, debug


def retrieval_metrics(ranked, gold):
    if not gold:
        return {key: None for key in ("Hit@1", "Hit@3", "MRR", "Recall@3")}
    first = next((i + 1 for i, code in enumerate(ranked) if code in gold), None)
    return {"Hit@1": float(bool(ranked) and ranked[0] in gold),
            "Hit@3": float(bool(set(ranked[:3]) & gold)),
            "MRR": 1 / first if first else 0.,
            "Recall@3": len(set(ranked[:3]) & gold) / len(gold)}


def calibrate_threshold(model, validation_rows):
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("Threshold calibration accepts validation rows only")
    samples = []
    for row in validation_rows:
        _, score, debug = ranked_codes(model, row["query_text"])
        samples.append({"query_id": row["query_id"], "score": score,
                        "has_answer": bool(gold_codes(row)), "all_oov": debug.get("all_oov", False)})
    if {s["has_answer"] for s in samples} != {True, False}:
        raise ValueError("Validation needs both answerable and unanswerable queries")
    values = sorted({s["score"] for s in samples})
    candidates = [float(np.nextafter(values[0], -np.inf))] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [float(np.nextafter(values[-1], np.inf))]
    best = None
    for threshold in candidates:
        accepted = [s["score"] >= threshold and not s["all_oov"] for s in samples]
        tpr = np.mean([a for a, s in zip(accepted, samples) if s["has_answer"]])
        tnr = np.mean([not a for a, s in zip(accepted, samples) if not s["has_answer"]])
        candidate = (float((tpr + tnr) / 2), float(tnr), threshold)
        if best is None or candidate > best:
            best = candidate
    return {"threshold": best[2], "validation_balanced_accuracy": best[0],
            "corpus_hash": model.fingerprint, "samples": samples,
            "manifest_sha256": hashlib.sha256((model.directory / "manifest.json").read_bytes()).hexdigest(),
            "selection": "Maximize balanced acceptance/rejection accuracy; ties favor rejection."}


def evaluate_model(model, rows):
    details = []
    for row in rows:
        gold = gold_codes(row)
        ranked, score, debug = ranked_codes(model, row["query_text"])
        response = model.retrieve(row["query_text"], top_k=3)
        for mode, returned, accepted in [
            ("retrieval_only", ranked, score >= model.threshold and not debug.get("all_oov", False)),
            ("end_to_end", list(dict.fromkeys(r["course_code"] for r in response["results"])), response["status"] == "ok")]:
            details.append({"Model": model.model_id, "mode": mode, "query_id": row["query_id"],
                            "category": row["query_category"], "has_answer": bool(gold), "accepted": bool(accepted),
                            "top_score": score, "status": response["status"] if mode == "end_to_end" else "ranked",
                            "top3": "|".join(returned[:3]), **retrieval_metrics(returned, gold)})
    return details


def summarize(details):
    def mean(field, predicate=lambda row: True):
        values = [r[field] for r in details if predicate(r) and r[field] is not None]
        return float(np.mean(values)) if values else None
    negative = [r for r in details if not r["has_answer"]]
    positive = [r for r in details if r["has_answer"]]
    ood = [r for r in negative if r["category"] == "out_of_domain"]
    return {**{k: mean(k) for k in ("Hit@1", "Hit@3", "MRR", "Recall@3")},
            "OOD_Accuracy": float(np.mean([not r["accepted"] for r in ood])) if ood else None,
            "False_Acceptance_Rate": float(np.mean([r["accepted"] for r in negative])) if negative else None,
            "False_Rejection_Rate": float(np.mean([not r["accepted"] for r in positive])) if positive else None,
            "queries": len(details), "answerable_queries": len(positive)}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
