"""RAG helper: search + safety guardrails + cited answer (no LLM yet)."""
import json
import re
from pathlib import Path

import numpy as np

# index folder: ya file chya sapekshe (notebook kinva Streamlit, kuthunhi chalel)
INDEX_DIR = Path(__file__).resolve().parent.parent / "rag" / "index"
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SCORE = 0.50          # 0.30 hota. Dev var 3 chukle prashna baghun vadhvla

DISCLAIMER = ("Educational information only, not medical advice. "
              "Please talk to a doctor or other health professional about your own health.")

# source file -> label. Page number fakt KDIGO saathi dakhavto
SOURCE_LABELS = {
    "CDC Basics.pdf": "CDC: Chronic Kidney Disease Basics",
    "CDC Risk Factors.pdf": "CDC: Risk Factors for Chronic Kidney Disease",
    "CDC Testing.pdf": "CDC: Testing for Chronic Kidney Disease",
    "CDC Living with CKD.pdf": "CDC: Living with Chronic Kidney Disease",
    "eGFR (NKF).pdf": "National Kidney Foundation: Estimated GFR (eGFR)",
    "CKD Stages G1-G5 (NKF).pdf": "National Kidney Foundation: Stages of CKD",
    "6-Step Guide (NKF).pdf": "National Kidney Foundation: 6-Step Guide to Protecting Kidney Health",
    "KDIGO 2024 Executive Summary.pdf": "KDIGO 2024 CKD Guideline, Executive Summary (Kidney Int 2024)",
}

# ---------- Guardrails ----------
# 1) personal prashna: "I", "my" asel tar (diagnosis / personal advice)
PERSONAL = re.compile(r"\b(i|my|mine|myself)\b(?!\.)|\bdiagnos(e|ing)\s+me\b")
# 2) treatment / dose / medicine nivadnyache prashna
TREATMENT = re.compile(
    r"\b(dose|dosage|dosing|prescribe|prescription|cure|how to treat|how do you treat|"
    r"treatment (for|of|options)|which (medicine|medication|drug)s?|"
    r"best (medicine|medication|drug|treatment)|what (medicine|medication|drug)s?)\b"
    r"|\bhow (much|many)\b[^?]*\b(take|taking|dose|mg|tablets?|pills?)\b")   # navin: "how much X can a person take"

REFUSAL = {
    "personal": ("I can't diagnose or give personal medical advice, so I can't answer questions "
                 "about your own symptoms, test results, or what you should do. "
                 "Please speak to a doctor. I can answer general questions about kidney health "
                 "from the documents, for example 'What is eGFR?'."),
    "treatment": ("I can't give treatment, medicine or dosing advice. Please speak to a doctor. "
                  "I can answer general questions about kidney health from the documents."),
}

_state = {}

def _load():
    """Chunks, embeddings ani model ekdach load karto, mag cache thevto."""
    if not _state:
        from sentence_transformers import SentenceTransformer
        with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
            _state["chunks"] = json.load(f)
        _state["emb"] = np.load(INDEX_DIR / "embeddings.npy")
        _state["model"] = SentenceTransformer(MODEL_NAME)
    return _state

def search(question, k=3):
    s = _load()
    q = s["model"].encode([question], normalize_embeddings=True)[0]   # prashna cha vector
    scores = s["emb"] @ q                                            # prattek chunk shi similarity
    top = scores.argsort()[::-1][:k]                                 # sarvat javal che k
    return [(float(scores[i]), s["chunks"][i]) for i in top]

def check_refusal(question):
    """'personal' / 'treatment' / None."""
    q = question.lower()
    if PERSONAL.search(q):
        return "personal"
    if TREATMENT.search(q):
        return "treatment"
    return None

def _label(chunk):
    label = SOURCE_LABELS.get(chunk["source"], chunk["source"])
    if chunk["source"].startswith("KDIGO"):
        label += f", PDF p.{chunk['page']}"
    return label

def answer(question, k=3, min_score=MIN_SCORE):
    reason = check_refusal(question)
    if reason:                                           # rule 1: diagnosis/treatment advice nahi
        return {"status": "refused", "text": REFUSAL[reason], "sources": [], "disclaimer": DISCLAIMER}

    hits = search(question, k=k)
    hits = [(s, c) for s, c in hits if s >= min_score]   # kami score che chunks kaadhto
    if not hits:                                         # rule 2: documents madhe nasel tar I don't know
        return {"status": "unknown",
                "text": "I don't know. The documents I have don't cover this question.",
                "sources": [], "disclaimer": DISCLAIMER}

    lines, sources = [], []
    for n, (score, c) in enumerate(hits, start=1):       # rule 3: prattek passage la source
        lines.append(f"[{n}] {c['text'][:350].strip()}...")
        sources.append(f"[{n}] {_label(c)} (match {score:.2f})")
    return {"status": "answered",
            "text": "Here are the most relevant passages from the documents:\n\n" + "\n\n".join(lines),
            "sources": sources, "disclaimer": DISCLAIMER}