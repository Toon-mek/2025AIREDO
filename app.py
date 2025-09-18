import os, streamlit as st, pandas as pd, numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.title("Laptop Recommender — Logistic Regression")

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)

MODEL_PATH = first_existing(["artifacts/logreg_pipeline.joblib", "logreg_pipeline.joblib"])
DATA_PATH  = first_existing(["data/logistics_regression.csv", "logistics_regression.csv"])

# --- Load data ---
df = pd.read_csv(DATA_PATH)

# --- Try to load trained pipeline; if it fails (pickle mismatch), retrain quickly in-app ---
pipe = None
try:
    pipe = load(MODEL_PATH)
except Exception as e:
    st.warning(f"Model file failed to load (env mismatch). Retraining inside app…\n{e}")
    # Build label from rules
    rg = pd.to_numeric(df.get("ram_gb"), errors="coerce")
    sd = pd.to_numeric(df.get("ssd"), errors="coerce")
    df["fit_cs"] = ((rg >= 8) & (sd >= 256)).astype(int)
    if df["fit_cs"].sum() == 0:
        df["fit_cs"] = ((rg >= 4) & (sd >= 128)).astype(int)

    X = df.drop(columns=["price_myr", "fit_cs"], errors="ignore")
    y = df["fit_cs"].astype(int)

    # minimal preprocess + model
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]).fit(X, y)

# --- Score & UI ---
X_all = df.drop(columns=["price_myr", "fit_cs"], errors="ignore")
df["p_fit"] = pipe.predict_proba(X_all)[:, 1]

budget = st.number_input("Budget (MYR)", 1000, 20000, 3000, 100)
k      = st.slider("Top-K", 1, 20, 10)
strict = st.checkbox("Strict (drop any missing in key fields)", True)

show = [c for c in ["brand","model","price_myr","ram_gb","ssd","processor_brand","processor_gnrtn","p_fit"] if c in df.columns]
view = df.query("price_myr <= @budget")[show]
if strict: view = view.dropna(how="any")

st.dataframe(view.sort_values("p_fit", ascending=False).head(k))

# (optional) show versions to confirm runtime
try:
    import sklearn, joblib, platform
    st.caption(f"Python {platform.python_version()} • sklearn {sklearn.__version__} • joblib {joblib.__version__}")
except Exception:
    pass
