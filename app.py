import os, streamlit as st, pandas as pd
from joblib import load

st.title("Laptop Recommender")

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)

model_path = first_existing(["artifacts/logreg_pipeline.joblib", "logreg_pipeline.joblib"])
data_path  = first_existing(["data/logistics_regression.csv", "logistics_regression.csv"])

# --- compatibility shim for sklearn pickle mismatch ---
try:
    from sklearn.compose import _column_transformer as _ct  # type: ignore
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):  # noqa: N801  (match expected name)
            pass
        _ct._RemainderColsList = _RemainderColsList  # inject expected symbol
except Exception:
    pass

# --- load model + data ---
pipe = load(model_path)
df   = pd.read_csv(data_path)

X = df.drop(columns=["price_myr"], errors="ignore")
df["p_fit"] = pipe.predict_proba(X)[:, 1]

budget = st.number_input("Budget (MYR)", 1000, 20000, 3000, 100)
k      = st.slider("Top-K", 1, 20, 10)
strict = st.checkbox("Strict (drop any missing in key fields)", True)

show = [c for c in ["brand","model","price_myr","ram_gb","ssd",
                    "processor_brand","processor_gnrtn","p_fit"] if c in df.columns]
view = df.query("price_myr <= @budget")[show]
if strict:
    view = view.dropna(how="any")

st.dataframe(view.sort_values("p_fit", ascending=False).head(k))

# optional: show versions
import sklearn, joblib
st.caption(f"Python runtime OK • sklearn {sklearn.__version__} • joblib {joblib.__version__}")
