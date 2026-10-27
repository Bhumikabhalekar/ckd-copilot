"""Evaluation helpers for the RAG assistant: run questions, confusion matrix, threshold sweep."""
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.rag import answer, search

LABELS = ["answered", "unknown", "refused"]


def run_eval(df):
    """Prattek prashna RAG la vicharto ani result gola karto."""
    rows = []
    for _, r in df.iterrows():
        out = answer(r["question"])                                  # RAG cha final result
        hits = search(r["question"], k=3)                            # top-3 (score cut nasatana)
        top_sources = [c["source"] for _, c in hits]
        expected = ([s.strip() for s in str(r["expected_source"]).split("|")]
                    if pd.notna(r["expected_source"]) else [])
        rows.append({
            "id": r["id"], "question": r["question"],
            "expected": r["expected_status"], "actual": out["status"],
            "top_score": round(hits[0][0], 3),                       # sarvat javal chunk cha score
            "source_hit": any(e in top_sources for e in expected),   # barobar document top-3 madhe aala ka
            "top_sources": top_sources,
        })
    return pd.DataFrame(rows)


def report(res, name):
    cm = confusion_matrix(res["expected"], res["actual"], labels=LABELS)
    print(f"=== {name} ===")
    print(pd.DataFrame(cm, index=[f"expected {l}" for l in LABELS],
                       columns=[f"got {l}" for l in LABELS]))
    ans = res[res["expected"] == "answered"]
    print("1) Hit@3 (answerable):", round(ans["source_hit"].mean(), 2))
    print("2) Wrongly not-answered:", round((ans["actual"] != "answered").mean(), 2))
    print("3) Correct 'I don't know':",
          round((res[res.expected == "unknown"]["actual"] == "unknown").mean(), 2))
    print("4) Refusal rate (unsafe):",
          round((res[res.expected == "refused"]["actual"] == "refused").mean(), 2))


def failures(res):
    bad = (res.expected != res.actual) | ((res.expected == "answered") & (~res.source_hit))
    return res[bad][["id", "question", "expected", "actual", "top_score", "top_sources"]]


def threshold_sweep(res, thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)):
    """Kontya threshold la kiti chuka (fakt answered/unknown prashnanvar)."""
    scope = res[res.expected.isin(["answered", "unknown"])]
    rows = []
    for t in thresholds:
        wrong_unknown = int(((scope.expected == "answered") & (scope.top_score < t)).sum())
        wrong_answered = int(((scope.expected == "unknown") & (scope.top_score >= t)).sum())
        rows.append({"threshold": t, "wrongly_unknown": wrong_unknown,
                     "wrongly_answered": wrong_answered, "total_errors": wrong_unknown + wrong_answered})
    return pd.DataFrame(rows)