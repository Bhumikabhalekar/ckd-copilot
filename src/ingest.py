"""PDF -> clean text -> sentence-aware chunks (offline build step, app la lagat nahi)."""
import re
from pathlib import Path

import pymupdf                                   # PDF vachnyasaathi (ligatures nit handle karto)

# PDF madhle ligature chinha -> normal letters (ﬁ -> fi)
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}

# repeat hone-vale faltu text (page header/footer, table titles)
NOISE_PATTERNS = [
    r"\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M .*? \| CDC",                 # CDC print header
    r"https://www\.cdc\.gov/\S+ \d+/\d+",                                     # CDC print footer
    r"A Levin et al\.: Executive summary of KDIGO 2024 CKD Guideline",        # KDIGO page header
    r"KDIGO executive conclusions",
    r"Kidney International \(2024\) 105, 684.701",                            # KDIGO page footer
    r"Table \d+ \| (?:\(Continued\) )?Recommendations and practice points from Chapters? [\d ]+(?:and \d+ )?"
    r"of the KDIGO 2024 Clinical Practice Guideline for\s+Evaluation and Management of CKD",   # KDIGO table titles
    r"\(Continued on following page\)",
    r"Table \d+ ?\| \(Continued\)",                                           # KDIGO "(Continued)" titles
]


def clean_text(text):
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)                      # ﬁ -> fi
    text = re.sub(r"\s+", " ", text)                       # newlines/extra spaces -> ek space
    for p in NOISE_PATTERNS:
        text = re.sub(p, " ", text, flags=re.DOTALL)       # faltu text kaadhto
    return re.sub(r"\s+", " ", text).strip()


def load_pages(pdf_path):
    """(page_no, clean_text) list. KDIGO madhe DISCLOSURE nantar (conflicts + references) kaapto."""
    pages, stop = [], False
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text()
            if "DISCLOSURE" in raw:
                raw, stop = raw.split("DISCLOSURE")[0], True
            txt = clean_text(raw)
            if txt:
                pages.append((i, txt))
            if stop:
                break
    return pages


def split_sentences(text):
    """Vakya sampla (. ! ?) ani pudhe capital letter / ( / [ asel tithe todto."""
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text) if s]


def make_chunks(text, max_chars=900, max_overlap=300):
    """Purna vakyanche chunks. Pudhcha chunk maage chya shevtchya vakyane suru hoto (overlap)."""
    pieces = []
    for s in split_sentences(text):
        while len(s) > max_chars:                          # khup lamb vakya: shabdanvar todto
            cut = s.rfind(" ", 0, max_chars)
            pieces.append(s[:cut])
            s = s[cut + 1:]
        pieces.append(s)
    chunks, cur, new = [], [], 0
    for s in pieces:
        if cur and len(" ".join(cur + [s])) > max_chars:
            if new:                                        # navin vakya astil tarach chunk kaadhto
                chunks.append(" ".join(cur))
                tail = cur[-1] if len(cur[-1]) <= max_overlap else None   # shevtcha vakya overlap mhanun
                cur, new = ([tail] if tail else []), 0
            if cur and len(" ".join(cur + [s])) > max_chars:   # overlap + navin vakya khup motha: overlap sodto
                cur = []
        cur.append(s)
        new += 1
    if new:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(docs_dir, allowed):
    """allowed = fakt ya PDF file names vaparayche."""
    out = []
    for pdf in sorted(Path(docs_dir).glob("*.pdf")):
        if pdf.name not in allowed:
            continue
        for page_no, txt in load_pages(pdf):
            for c in make_chunks(txt):
                if len(c) >= 80:                           # 80 characters peksha lahan tukde (page number etc.) soDto
                    out.append({"source": pdf.name, "page": page_no, "text": c})
    return out