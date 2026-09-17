"""
Document parsing: extracts text per page from PDFs, keeping source metadata
(filename, page number) attached to every chunk downstream — this is what
lets the pipeline cite *which* document and page an answer came from.
"""
import os
from pypdf import PdfReader


def parse_pdf(path: str) -> list[dict]:
    """Returns a list of {source, page, text} dicts, one per non-empty page."""
    reader = PdfReader(path)
    filename = os.path.basename(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"source": filename, "page": i + 1, "text": text})
    return pages


def parse_directory(dir_path: str) -> list[dict]:
    """Parses every PDF in a directory, returns the combined page list."""
    all_pages = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".pdf"):
            all_pages.extend(parse_pdf(os.path.join(dir_path, fname)))
    return all_pages


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(os.path.dirname(here), "data", "sample_docs")
    pages = parse_directory(docs_dir)
    print(f"Parsed {len(pages)} pages from {docs_dir}")
    for p in pages[:2]:
        print(f"--- {p['source']} p{p['page']} ---")
        print(p["text"][:200], "...\n")
