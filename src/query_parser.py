from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .preprocessing import CODE_RE, correct_typos, normalize_codes, semantic_tokens
from .utils import ROOT, read_json


@dataclass
class ParsedQuery:
    original: str
    intent: str = "GENERAL_SEARCH"
    requested_field: str | None = None
    course_code: str | None = None
    course_title: str | None = None
    year: str | None = None
    term: str | None = None
    search_query: str = ""
    corrections: dict | None = None

    def as_dict(self):
        return asdict(self)


class QueryParser:
    def __init__(self, courses):
        self.courses = courses
        self.aliases = read_json(ROOT / "data/queries/aliases.json")
        self.vocabulary = set(semantic_tokens(" ".join(c["search_text"] for c in courses)))
        self.vocabulary.update("course courses teaches teaching taught credits credit contact hours details prerequisites teachers where which what please about learn learning study studies subject subjects covers discusses explain explains protecting computer systems fourth third second first tomorrow semantic natural language programming".split())

    def parse(self, query):
        p = ParsedQuery(original=query, corrections={})
        text = normalize_codes(query.strip())
        if not text:
            p.intent = "EMPTY"
            return p
        # Unsupported fields take precedence over otherwise answerable code lookups.
        if re.search(r"\b(prerequisites?|teachers?|instructors?|professors?|schedules?|class time|classroom|tuition|fees?|enroll|admission)\b|\bwho\s+(?:teaches|will teach|is teaching)\b", text, re.I):
            p.intent, p.requested_field = "UNSUPPORTED_FIELD", "unavailable"
            return p
        if re.search(r"\b(weather|president|football|world cup|cook|pasta|stock price|horoscope|celebrity)\b", text, re.I):
            p.intent = "OUT_OF_SCOPE"
            return p
        code = CODE_RE.search(text)
        if code:
            p.course_code = f"{code[1].upper()} {code[2]}"
        for alias, title in self.aliases.items():
            text = re.sub(rf"\b{re.escape(alias)}\b", title, text, flags=re.I)
        text, p.corrections = correct_typos(text, self.vocabulary)
        lower = text.lower()
        for c in sorted(self.courses, key=lambda c: -len(c["course_title"])):
            if c["course_title"].lower() in lower:
                p.course_title = c["course_title"]
                break
        numbers = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                   "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "1": 1, "2": 2, "3": 3, "4": 4}
        ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        for field, maximum in [("year", 4), ("term", 2)]:
            m = re.search(rf"\b(first|second|third|fourth|[1-4](?:st|nd|rd|th)?)\s+{field}\b|\b{field}\s+([1-4])\b", lower)
            if m and numbers[m[1] or m[2]] <= maximum:
                setattr(p, field, f"{ordinals[numbers[m[1] or m[2]]]} {field.title()}")
        if re.search(r"\bcredits?\b", lower):
            p.intent, p.requested_field = "CREDIT_QUERY", "credits"
        elif re.search(r"\b(contact|hours?)\b", lower):
            p.intent, p.requested_field = "CONTACT_HOUR_QUERY", "contact_hours"
        elif re.search(r"\b(lab|laboratory|sessional)\b", lower):
            p.intent = "LAB_QUERY"
            p.requested_field = "parent_course"
        elif p.year or p.term or re.search(r"\b(year|term|semester)\b", lower):
            p.intent = "YEAR_TERM_QUERY"
            p.requested_field = "year_term"
        elif p.course_code:
            p.intent = "COURSE_CODE_QUERY" if normalize_codes(query.strip()).upper() == p.course_code else "COURSE_DETAILS"
        elif re.search(r"\b(about|details|describe)\b", lower) or p.course_title:
            p.intent = "COURSE_DETAILS"
        elif re.search(r"\b(teaches|teach|taught|learn|covers|cover|discusses|topic|course)\b", lower):
            p.intent = "TOPIC_SEARCH"
        search = re.sub(r"\b(what|which|where|when|how|many|much|is|are|the|a|an|do|does|we|i|can|in|on|of|for|with|have|has|right|please|tell|me|about|course|courses|teaches|teach|taught|learn|learning about|covers|cover|discusses|credits?|contact|hours?|details)\b", " ", lower)
        search = re.sub(r"[^a-z0-9+#\s-]", " ", search)
        p.search_query = p.course_title or " ".join(search.split()) or lower
        tokens = semantic_tokens(p.search_query)
        if not p.course_code and not (p.year or p.term) and (not tokens or all(not re.search(r"[aeiouy]", t) for t in tokens) or re.fullmatch(r"(?:asdf|qwer|zx|xyz|blah|\s)+", lower)):
            p.intent = "GIBBERISH"
        return p
