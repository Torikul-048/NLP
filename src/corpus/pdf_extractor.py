"""Extract in PDF reading order, keeping physical (one-based) source pages."""
from pathlib import Path


def extract_pages(pdf_path):
    import pymupdf
    with pymupdf.open(Path(pdf_path)) as document:
        return [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(document)]
