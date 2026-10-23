"""Classifier helper: model load, input row, prediction, SHAP explanation."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

# project root: ya file chya sapekshe, mhanun app kuthunhi chalo
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "ckd_pipeline.joblib"
DATA_PATH = ROOT / "data" / "processed" / "ckd_clean.csv"

NUM_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
            "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
CAT_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
ALL_COLS = NUM_COLS + CAT_COLS

# samajnare naave (UCI dataset chya units pramane)
LABELS = {
    "age": "Age (years)", "bp": "Blood pressure (mm Hg)",
    "sg": "Urine specific gravity", "al": "Urine albumin (0-5)", "su": "Urine sugar (0-5)",
    "bgr": "Random blood glucose (mg/dL)", "bu": "Blood urea (mg/dL)",
    "sc": "Serum creatinine (mg/dL)", "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)",
    "hemo": "Hemoglobin (g/dL)", "pcv": "Packed cell volume", "wbcc": "White blood cell count (cells/cumm)",
    "rbcc": "Red blood cell count (millions/cmm)",
    "rbc": "Red blood cells in urine", "pc": "Pus cells in urine", "pcc": "Pus cell clumps",
    "ba": "Bacteria in urine", "htn": "Hypertension", "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease", "appet": "Appetite", "pe": "Pedal edema (leg swelling)",
    "ane": "Anemia",
}


def load_pipeline():
    return joblib.load(MODEL_PATH)               # preprocessing + model ekach file madhe


def load_data():
    return pd.read_csv(DATA_PATH)                # cleaned data (public UCI dataset)


def input_options(df):
    """Dataset madhun dropdown che options ani number che min/max kaadhto (hardcode nahi)."""
    options, ranges = {}, {}
    for c in ["sg", "al", "su"] + CAT_COLS:
        vals = sorted(df[c].dropna().unique())
        if c in ("al", "su"):
            vals = [int(v) for v in vals]
        elif c == "sg":
            vals = [float(v) for v in vals]
        options[c] = vals
    for c in NUM_COLS:
        if c not in ("sg", "al", "su"):
            ranges[c] = (float(df[c].min()), float(df[c].max()))
    return options, ranges


def pick_examples(df):
    """Prattek class madhun sarvat kami missing values asleli pahili row (demo example)."""
    d = df.assign(_missing=df.isna().sum(axis=1))
    ex = {}
    for cls in ["ckd", "notckd"]:
        row = d[d["class"] == cls].sort_values("_missing", kind="stable").iloc[0]
        ex[cls] = row.drop(["class", "_missing"]).to_dict()
    return ex


def build_explainer(pipe, df):
    """SHAP explainer: background = same train split (random_state=42) jo model shikla."""
    X = df.drop(columns=["class"])
    y = (df["class"] == "ckd").astype(int)
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    prep = pipe.named_steps["prep"]
    background = prep.transform(X_train)
    # max_samples = sagle train rows, mhanje random subsampling nahi, result nehmi same
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(pipe.named_steps["model"], masker)
    return explainer, list(prep.get_feature_names_out())


def make_row(values):
    """User ne dilelya values (nasel tar NaN) pasun 1-row DataFrame. NaN pipeline bharto."""
    row = {c: (np.nan if values.get(c) is None else values[c]) for c in ALL_COLS}
    return pd.DataFrame([row], columns=ALL_COLS)


def predict_score(pipe, row):
    return float(pipe.predict_proba(row)[0, 1])  # model cha CKD score (0 te 1)


def explain_row(pipe, explainer, feature_names, row):
    """Prattek original feature saathi SHAP. One-hot columns (jase htn_yes + htn_no) ekatra."""
    shap_vals = explainer(pipe.named_steps["prep"].transform(row)).values[0]
    contrib = {}
    for name, v in zip(feature_names, shap_vals):
        kind, rest = name.split("__", 1)                      # 'num__hemo' / 'cat__htn_yes'
        col = rest if kind == "num" else rest.split("_", 1)[0]
        contrib[col] = contrib.get(col, 0.0) + float(v)
    table = pd.DataFrame({
        "feature": list(contrib),
        "label": [LABELS[c] for c in contrib],
        "value": [row.iloc[0][c] for c in contrib],
        "shap": list(contrib.values()),
    })
    table["provided"] = table["value"].notna()
    return table.sort_values("shap", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)