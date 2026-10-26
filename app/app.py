import sys
from pathlib import Path

# "streamlit run app/app.py" chalvla ki project root path madhe nasto, mhanun add karto
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import classifier as clf
from src.rag import answer, warm_up

st.set_page_config(page_title="CKD Risk & Clinical Copilot", page_icon="🩺", layout="wide")

DISCLAIMER = ("Educational project, not medical advice. This app cannot diagnose any condition "
              "or recommend treatment. Please talk to a doctor about your own health.")


@st.cache_resource                       # model, SHAP explainer, data fakt ekda load hotat
def load_assets():
    pipe = clf.load_pipeline()
    df = clf.load_data()
    explainer, names = clf.build_explainer(pipe, df)
    options, ranges = clf.input_options(df)
    return pipe, explainer, names, options, ranges, clf.pick_examples(df)

@st.cache_resource                       # ekdach chalto, parat parat nahi
def load_rag():
    warm_up()                            # RAG model + index load
    return True


def is_missing(v):
    return v is None or bool(pd.isna(v))       # None, NaN, pd.NA sagle missing


def default_index(opts, default):
    """Dropdown madhe example cha value konta aahe (nasel tar None = 'Not provided')."""
    if is_missing(default):
        return None
    if isinstance(opts[0], int):         # al, su: 0-5 whole numbers
        default = int(default)
    return opts.index(default) if default in opts else None


def show(v):
    """Chart label saathi value cha text."""
    if is_missing(v):
        return "not provided"
    return f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)


def contribution_chart(table, n=8):
    top = table.head(n).iloc[::-1]       # mothe contributions varti disavet
    labels = [f"{r.label}: {show(r.value)}" for r in top.itertuples()]
    colors = ["#d62728" if s > 0 else "#1f77b4" for s in top["shap"]]   # laal = CKD kade, nila = door
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, top["shap"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Pushes the score away from CKD  <-  0  ->  toward CKD")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Tab 1: classifier
def classifier_tab():
    try:
        pipe, explainer, names, options, ranges, examples = load_assets()
    except FileNotFoundError as e:
        st.error(f"Model or data file not found: {e}")
        return

    st.write("Enter lab values and history. **Leave a field empty if it was not tested**: "
             "the model fills it with the typical value from its training data.")
    examples_map = {"Empty form": {},
                    "Example: CKD patient (UCI dataset)": examples["ckd"],
                    "Example: non-CKD patient (UCI dataset)": examples["notckd"]}
    choice = st.selectbox("Start from", list(examples_map))
    ex = examples_map[choice]

    values = {}
    with st.form("patient_form"):
        cols = st.columns(3)
        for i, c in enumerate(clf.ALL_COLS):
            box = cols[i % 3]
            key = f"{c}|{choice}"        # key madhe choice, mhanje example badalla ki values reset hotat
            default = None if is_missing(ex.get(c)) else ex.get(c)
            if c in options:             # dropdown (categories ani 0-5 levels)
                opts = options[c]
                values[c] = box.selectbox(clf.LABELS[c], opts, index=default_index(opts, default),
                                          key=key, placeholder="Not provided")
            else:                        # number
                lo, hi = ranges[c]
                values[c] = box.number_input(clf.LABELS[c], min_value=lo, max_value=hi, value=default,
                                             step=1.0 if hi - lo > 100 else 0.1, key=key,
                                             placeholder="Not provided")
        submitted = st.form_submit_button("Calculate")

    if not submitted:
        return
    row = clf.make_row(values)           # nasel te NaN, pipeline bharto
    score = clf.predict_score(pipe, row)
    n_given = sum(v is not None for v in values.values())

    left, right = st.columns([1, 2])
    with left:
        st.metric("Model score for a CKD pattern", f"{score * 100:.0f}%")
        st.progress(score)
        st.caption(f"{n_given} of {len(clf.ALL_COLS)} values provided. This is a model score, not a "
                   "diagnosis. The training data had 62.5% CKD patients, far more than in real life, "
                   "so the score overstates real-world risk.")
        if n_given < 8:
            st.warning("Very few values entered. The score mostly reflects default values.")
    with right:
        table = clf.explain_row(pipe, explainer, names, row)
        fig = contribution_chart(table)
        st.pyplot(fig)
        plt.close(fig)                   # memory free karto
        st.caption("Red bars push the score toward CKD, blue bars away from it. "
                   "This shows what the model used. It does not prove a cause.")


# ------------------------------------------------------------------ Tab 2: RAG
def rag_tab():
    st.write("Ask a general question about kidney health. Answers come only from CDC and KDIGO "
             "documents, with sources. Personal medical questions are refused.")
    with st.form("rag_form"):
        question = st.text_input("Your question", placeholder="What is eGFR?")
        asked = st.form_submit_button("Ask")
    if not (asked and question.strip()):
        return
    try:
        with st.spinner("Searching the documents..."):
            out = answer(question.strip())
    except FileNotFoundError:
        st.error("Document index not found. Build it first (notebooks 03 and 05).")
        return
    if out["status"] == "answered":
        st.markdown(out["text"])
        st.markdown("**Sources**")
        for s in out["sources"]:
            st.markdown(f"- {s}")
    elif out["status"] == "unknown":
        st.info(out["text"])
    else:
        st.warning(out["text"])
    st.caption(out["disclaimer"])


# ------------------------------------------------------------------ Tab 3: About
def about_tab():
    st.markdown("""
**What this is:** a learning project with two parts. A logistic regression model trained on the UCI
Chronic Kidney Disease dataset, and a retrieval assistant over public kidney-health documents.

**Limitations (please read):**
- The dataset has only 400 patients from one hospital and is unusually clean, so near-perfect scores
  do not mean the model works in a real clinic. It has **not** been clinically validated.
- The score is not a probability for a real person. The explanation shows what the model relied on,
  not what caused anything.
- The document assistant can be wrong or miss relevant text. It may answer questions the documents only
  partly cover, and its refusal rules can miss unusual wording.

**Sources:** CDC and the KDIGO 2024 CKD Guideline (executive summary).
""")


# ------------------------------------------------------------------ Page
st.title("🩺 Explainable CKD Risk & Clinical Copilot")
st.warning(DISCLAIMER)
tab_model, tab_rag, tab_about = st.tabs(["Risk predictor", "Ask the documents", "About & limitations"])
with tab_model:
    classifier_tab()
with tab_rag:
    rag_tab()
with tab_about:
    about_tab()
# page display zalyavar model load karto, mhanje pahila prashna lagech yeto
with st.spinner("Loading the search model in the background (first time only)..."):
    load_rag()