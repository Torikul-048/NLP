from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from .pdf_extractor import extract_pages
from ..utils import write_json

HEADING = re.compile(r"^([A-Z]{2,5})\s+(\d{4})(?:\s*:\s*|\s+)(.+)$")
YEAR_TERM = re.compile(r"([1-4](?:st|nd|rd|th))\s*Year\s*([12](?:st|nd))\s*Term", re.I)
OPTIONAL = re.compile(r"Optional\s*-\s*(III|II|I)\b", re.I)


def parse_pages(pages):
    records, current = [], None
    year, term, group = None, None, None
    summary = False
    summary_lines = []

    def finish():
        nonlocal current
        if current is None:
            return
        raw = "\n".join(current.pop("lines"))
        credit = re.search(r"Credits?\s*:\s*(\d+(?:\.\d+)?)", raw, re.I)
        contact = re.search(r"Contact\s*Hours?\s*:\s*([\dLPT+/.\s]+?)(?:Hrs|Hours|$)", raw, re.I | re.M)
        description = re.sub(r"Credits?\s*:\s*\d+(?:\.\d+)?", "", raw, flags=re.I)
        description = re.sub(r"Contact\s*Hours?\s*:[^\n]*", "", description, flags=re.I)
        description = " ".join(description.split())
        title = current["course_title"]
        parent = re.search(r"Laboratory works? based on\s+([A-Z]{2,5})\s*(\d{4})", description, re.I)
        current.update(credits=float(credit[1]) if credit else None,
                       contact_hours=re.sub(r"\s+", "", contact[1]) if contact else None,
                       description=description, raw_text=current.pop("heading") + "\n" + raw,
                       parent_course=f"{parent[1].upper()} {parent[2]}" if parent else None,
                       course_type="Laboratory" if parent or "laboratory" in title.lower() else
                                   "Project" if re.search(r"project|thesis|seminar|training", title, re.I) else "Theory",
                       generic_lab=bool(parent and len(description.split()) < 12),
                       search_text=f"{current['course_code']} {title} {title} {description}",
                       record_id=f"{current['course_code'].replace(' ', '_')}_{current['year']}_{current['term']}",
                       source_conflicts=[])
        records.append(current)
        current = None

    for page in pages:
        for raw_line in page["text"].splitlines():
            line = raw_line.strip()
            if not line or re.fullmatch(r"\d{1,2}", line):
                continue
            if re.match(r"(?:Summary|Syllabus) of", line, re.I):
                finish()
                summary = line.lower().startswith("summary")
                yt, opt = YEAR_TERM.search(line), OPTIONAL.search(line)
                if yt:
                    year, term = f"{yt[1]} Year", f"{yt[2]} Term"
                    group = None
                elif opt:
                    group = "Optional-" + opt[1].upper()
                continue
            if summary:
                summary_lines.append((line, year, term, group, page["page"]))
                continue
            match = HEADING.match(line)
            if match:
                finish()
                current = {"course_code": f"{match[1]} {match[2]}", "course_title": " ".join(match[3].split()),
                           "year": year, "term": term, "optional_group": group, "source_page": page["page"],
                           "source_pages": [page["page"]], "heading": line, "lines": []}
            elif current:
                current["lines"].append(line)
                if page["page"] not in current["source_pages"]:
                    current["source_pages"].append(page["page"])
    finish()
    if not records:
        raise ValueError("No course headings found. An image-only PDF needs OCR before parsing.")
    keys = [r["record_id"] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate code/year/term records detected; inspect the source PDF.")
    return records


def audit_summary_tables(pdf_path, courses):
    """Read table columns spatially; retain, rather than overwrite, contradictory values."""
    import pymupdf
    from fractions import Fraction
    year = term = None
    in_summary = False
    theory_x = practical_x = credit_x = None
    with pymupdf.open(pdf_path) as document:
        for page_no, page in enumerate(document, 1):
            rows = []
            for word in sorted(page.get_text("words"), key=lambda w: (round(w[1]), w[0])):
                if rows and abs(rows[-1][0] - word[1]) < 2:
                    rows[-1][1].append(word)
                else:
                    rows.append((word[1], [word]))
            for _, words in rows:
                words.sort(key=lambda w: w[0])
                line = " ".join(w[4] for w in words)
                if "Summary of" in line or "Syllabus of" in line:
                    in_summary = "Summary of" in line
                    yt = YEAR_TERM.search(line)
                    if yt:
                        year, term = f"{yt[1]} Year", f"{yt[2]} Term"
                if not in_summary:
                    continue
                for w in words:
                    if w[4] == "L": theory_x = (w[0] + w[2]) / 2
                    if w[4] == "P": practical_x = (w[0] + w[2]) / 2
                    if w[4] == "Credit": credit_x = (w[0] + w[2]) / 2
                match = re.match(r"([A-Z]{2,5})\s+(\d{4})\b", line)
                if not match or None in (theory_x, practical_x, credit_x):
                    continue
                vals = []
                for x in (theory_x, practical_x, credit_x):
                    cell = "".join(w[4] for w in words if abs((w[0] + w[2]) / 2 - x) < 15)
                    try:
                        vals.append(float(Fraction(cell)) if cell else 0.)
                    except (ValueError, ZeroDivisionError):
                        vals.append(None)
                code = f"{match[1]} {match[2]}"
                for c in courses:
                    if (c["course_code"], c["year"], c["term"]) != (code, year, term):
                        continue
                    c["summary_metadata"] = dict(theory_hours=vals[0], practical_hours=vals[1], credits=vals[2], source_page=page_no)
                    if vals[2] is not None and vals[2] != c["credits"]:
                        c["source_conflicts"].append({"field": "credits", "syllabus": c["credits"], "summary": vals[2], "summary_page": page_no})
                    contact = re.match(r"([\d./]+)L\+(?:[\d./]+T\+)?([\d./]+)P", c["contact_hours"] or "")
                    if contact and None not in vals[:2] and [float(Fraction(contact[1])), float(Fraction(contact[2]))] != vals[:2]:
                        c["source_conflicts"].append({"field": "contact_hours", "syllabus": c["contact_hours"],
                                                     "summary": f"{vals[0]:g}L+{vals[1]:g}P", "summary_page": page_no})


def build_corpus(pdf_path, output):
    pdf_path, output = Path(pdf_path), Path(output)
    pages = extract_pages(pdf_path)
    courses = parse_pages(pages)
    audit_summary_tables(pdf_path, courses)
    write_json(output / "courses.json", courses)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "courses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(courses[0]))
        writer.writeheader()
        writer.writerows(courses)
    stats = {"source_pdf": pdf_path.name, "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
             "pages": len(pages), "records": len(courses), "unique_course_codes": len({c['course_code'] for c in courses}),
             "by_type": dict(Counter(c["course_type"] for c in courses)),
             "by_optional_group": dict(Counter(c["optional_group"] or "Compulsory" for c in courses)),
             "missing_credits": [c["record_id"] for c in courses if c["credits"] is None],
             "missing_contact_hours": [c["record_id"] for c in courses if c["contact_hours"] is None],
             "source_conflicts": [{"course_code": c["course_code"], "conflicts": c["source_conflicts"]} for c in courses if c["source_conflicts"]]}
    write_json(output / "corpus_statistics.json", stats)
    return stats
