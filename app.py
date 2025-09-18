import os, streamlit as st, pandas as pd
from joblib import load

st.title("Laptop Recommender — Logistic Regression")

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these exist: {paths}")

model_path = first_existing(["artifacts/logreg_pipeline.joblib", "logreg_pipeline.joblib"])
data_path  = first_existing(["data/logistics_regression.csv", "logistics_regression.csv"])

pipe = load(model_path)
df   = pd.read_csv(data_path)

X = df.drop(columns=["price_myr"], errors="ignore")
df["p_fit"] = pipe.predict_proba(X)[:, 1]

budget = st.number_input("Budget (MYR)", 1000, 20000, 3000, 100)
k      = st.slider("Top-K", 1, 20, 10)
strict = st.checkbox("Strict (drop any missing in key fields)", True)

show = [c for c in ["brand","model","price_myr","ram_gb","ssd","processor_brand","processor_gnrtn","p_fit"] if c in df.columns]
view = df.query("price_myr <= @budget")[show]
if strict: view = view.dropna(how="any")

st.dataframe(view.sort_values("p_fit", ascending=False).head(k))
