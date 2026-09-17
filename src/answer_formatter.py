import re

NO_INFO = "The supplied curriculum does not provide this information."
MESSAGES = {
    "empty": "Please enter an academic query.",
    "out_of_scope": "This query appears to be outside the scope of the supplied academic curriculum.",
    "gibberish": "No meaningful academic query could be identified.",
    "no_match": "No sufficiently relevant information was found in the supplied curriculum.",
    "insufficient_representation": "Insufficient semantic representation: no query tokens are covered by pretrained Word2Vec.",
    "unsupported": NO_INFO,
}


def excerpt(course, query):
    sentences = re.split(r"(?<=[.;])\s+", course.get("description", ""))
    terms = set(re.findall(r"\w+", query.lower()))
    return max(sentences, key=lambda s: len(terms & set(re.findall(r"\w+", s.lower()))), default="")


def format_answer(parsed, results, status):
    if status in MESSAGES:
        return MESSAGES[status]
    if not results:
        return NO_INFO
    c = results[0]
    name = f"{c['course_code']} — {c['course_title']}"
    if parsed.intent == "CREDIT_QUERY":
        lines = []
        for r in results:
            if r.get("credits") is None:
                lines.append(NO_INFO)
                continue
            line = f"{r['course_code']} — {r['course_title']} has {r['credits']:.1f} credits"
            line += f" ({r['year']}, {r['term']})." if len(results) > 1 else "."
            for conflict in r.get("source_conflicts", []):
                if conflict["field"] == "credits":
                    line += f" Source conflict: detailed syllabus states {conflict['syllabus']}; summary table on page {conflict['summary_page']} states {conflict['summary']}."
            lines.append(line)
        return "\n".join(lines)
    if parsed.intent == "CONTACT_HOUR_QUERY":
        lines = []
        for r in results:
            line = f"{r['course_code']} — {r['course_title']}: {r.get('contact_hours') or 'Not specified'} contact hours per week ({r['term']})."
            for conflict in r.get("source_conflicts", []):
                if conflict["field"] == "contact_hours":
                    line += f" Source conflict: summary table on page {conflict['summary_page']} states {conflict['summary']}."
            lines.append(line)
        return "\n".join(lines)
    if parsed.intent == "LAB_QUERY":
        return "Laboratory / sessional records in the supplied curriculum:\n" + "\n".join(f"• {r['course_code']} — {r['course_title']}" for r in results)
    if parsed.intent == "YEAR_TERM_QUERY":
        return "\n".join(f"• {r['course_code']} — {r['course_title']} ({r.get('year') or 'Year not specified'}, {r.get('term') or 'term not specified'})" for r in results)
    if parsed.intent in {"COURSE_DETAILS", "COURSE_CODE_QUERY"}:
        return f"{name}\nCredits: {c.get('credits')}\nYear: {c.get('year') or 'Not specified'}\nTerm: {c.get('term') or 'Not specified'}\nTopics: {c.get('description') or 'Not specified'}"
    label = "Matching courses" if len(results) > 1 else "Best matching course"
    return f"{label}:\n" + "\n\n".join(f"{r['course_code']} — {r['course_title']}\nRelevant curriculum content: {r['matched_excerpt']}" for r in results)
