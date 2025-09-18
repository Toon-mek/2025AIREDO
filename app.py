# app.py — train-on-start, no joblib, fast & version-proof
import os, streamlit as st, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Laptop Recommender")
st.title("Laptop Recommender — Logistic Regression")

@st.cache_data
def load_data():
    for p in ["data/logistics_regression.csv", "logistics_regression.csv"]:
        if os.path.exists(p): return pd.read_csv(p)
    raise FileNotFoundError("CSV not found in 'data/' or repo root.")

@st.cache_resource
def build_pipe(df):
    # label from RAM/SSD rule
    rg = pd.to_numeric(df.get("ram_gb"), errors="coerce")
    sd = pd.to_numeric(df.get("ssd"),    errors="coerce")
    df = df.copy()
    df["fit_cs"] = ((rg >= 8) & (sd >= 256)).astype(int)
    if df["fit_cs"].sum() == 0:
        df["fit_cs"] = ((rg >= 4) & (sd >= 128)).astype(int)

    X = df.drop(columns=["price_myr","fit_cs"], errors="ignore")
    y = df["fit_cs"].astype(int)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    pipe.fit(X, y)
    return pipe

df = load_data()
pipe = build_pipe(df)

X_all = df.drop(columns=["price_myr","fit_cs"], errors="ignore")
df = df.assign(p_fit=pipe.predict_proba(X_all)[:, 1])

# UI
budget = st.number_input("Budget (MYR)", 1000, 20000, 3000, 100)
k      = st.slider("Top-K", 1, 20, 10)
strict = st.checkbox("Strict (drop rows with any missing key fields)", True)

show = [c for c in ["brand","model","price_myr","ram_gb","ssd","processor_brand","processor_gnrtn","p_fit"] if c in df.columns]
view = df.query("price_myr <= @budget")[show]
if strict: view = view.dropna(how="any")

st.subheader("Recommendations")
st.dataframe(view.sort_values("p_fit", ascending=False).head(k), use_container_width=True)
