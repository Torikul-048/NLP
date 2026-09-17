"""Dataset validation and deterministic negative sampling; evaluation is never read by training."""
import csv
import random
import re
from collections import defaultdict

import numpy as np


def read_queries(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def gold_codes(row):
    return {x.strip() for x in (row.get("gold_course_code") or row.get("positive_course_code") or "").split("|") if x.strip()}


def validate_datasets(relevance, evaluation, courses):
    codes = {c["course_code"] for c in courses}
    seen_ids, seen_queries, groups = set(), {}, {}
    for row in relevance + evaluation:
        if row["query_id"] in seen_ids:
            raise ValueError("Duplicate query ID: " + row["query_id"])
        seen_ids.add(row["query_id"])
        split = row.get("split", "test")
        if split not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        text = re.sub(r"\W+", " ", row["query_text"].lower()).strip()
        if text in seen_queries and seen_queries[text] != split:
            raise ValueError("Exact query leakage across splits")
        seen_queries[text] = split
        group = row["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError("Paraphrase group leakage across splits: " + group)
        groups[group] = split
        if gold_codes(row) - codes:
            raise ValueError(f"Unknown gold course: {row}")
        if "has_answer" in row and (row["has_answer"].lower() == "true") != bool(gold_codes(row)):
            raise ValueError("has_answer disagrees with gold codes")
    return {"train": sum(r["split"] == "train" for r in relevance),
            "validation": sum(r["split"] == "validation" for r in relevance), "test": len(evaluation),
            "note": "Exact duplicates and declared groups checked; semantic paraphrase independence requires human review."}


def make_pairs(rows, courses, lexical_model, seed=42):
    rng = random.Random(seed)
    pairs = []
    for row in rows:
        gold = gold_codes(row)
        positives = [i for i, c in enumerate(courses) if c["course_code"] in gold]
        eligible = [i for i, c in enumerate(courses) if c["course_code"] not in gold and not c.get("generic_lab")]
        query = lexical_model.parser.parse(row["query_text"]).search_query or row["query_text"]
        if positives:
            scores, _ = lexical_model.score(query)
            hard = sorted(eligible, key=lambda i: -scores[i])[:3]
            rest = [i for i in eligible if i not in hard]
            negatives = [(i, "hard") for i in hard] + [(i, "random") for i in rng.sample(rest, min(2, len(rest)))]
        else:
            negatives = [(i, "out_of_domain") for i in rng.sample(eligible, min(5, len(eligible)))]
        for i, kind in [(i, "positive") for i in positives] + negatives:
            pairs.append({"query_id": row["query_id"], "query": query, "document_index": i,
                          "course_code": courses[i]["course_code"], "label": int(kind == "positive"), "sampling": kind})
    return pairs
