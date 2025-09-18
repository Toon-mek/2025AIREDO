# app.py — Minimal interactive Logistic Recommender (no plots, no joblib)
import os, numpy as np, pandas as pd, streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("Laptop Recommender (Logistic)")

@st.cache_data
def load_data():
    for p in ["data/logistics_regression.csv", "logistics_regression.csv"]:
        if os.path.exists(p): return pd.read_csv(p)
    raise FileNotFoundError("CSV not found in 'data/' or repo root.")

@st.cache_resource
def train_pipe(df, ram_rule: int, ssd_rule: int, balanced: bool, C: float):
    rg = pd.to_numeric(df.get("ram_gb"), errors="coerce")
    sd = pd.to_numeric(df.get("ssd"),    errors="coerce")
    y  = ((rg >= ram_rule) & (sd >= ssd_rule)).astype(int)
    X  = df.drop(columns=["price_myr"], errors="ignore")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre),
                     ("lr",  LogisticRegression(max_iter=1000,
                                               class_weight=("balanced" if balanced else None),
                                               C=C, random_state=42))])
    pipe.fit(X, y)
    return pipe

# ----- Data & sidebar -----
# ----- Data & sidebar -----
df = load_data()

with st.sidebar:
    st.header("Your preferences")
    budget = st.number_input("Budget (MYR)", 1000, 20000, 3000, 100)
    top_k  = st.slider("Top-K", 1, 30, 10)

    st.divider()
    st.caption("Label rule (what counts as a 'fit')")
    ram_rule = st.select_slider("RAM ≥ (GB)", options=[4,8,12,16,24,32], value=8)
    ssd_rule = st.select_slider("SSD ≥ (GB)", options=[128,256,512,1024], value=256)

    st.caption("Optional filters")
    # 1) Brand multiselect
    brands_all = sorted(df.get("brand", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    brand_sel = st.multiselect("Brand", brands_all, default=[])

    # 2) CPU options depend on brand selection
    sub = df[df["brand"].isin(brand_sel)] if brand_sel else df
    cpu_all = sorted(sub.get("processor_brand", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    cpu_sel = st.multiselect("CPU brand", cpu_all, default=[])

    st.divider()
    balanced = st.checkbox("class_weight='balanced'", True)
    C = st.select_slider("Regularization C", options=[0.5,1.0,2.0,5.0], value=1.0)

# ----- Train once, score all -----
pipe = train_pipe(df, ram_rule, ssd_rule, balanced, C)
X_all = df.drop(columns=["price_myr"], errors="ignore")
df = df.assign(p_fit = pipe.predict_proba(X_all)[:, 1])

# ----- Apply filters (brand first → cpu subset) -----
view = df[(df["price_myr"] <= budget)].copy()
if brand_sel:
    view = view[view["brand"].isin(brand_sel)]
if cpu_sel and "processor_brand" in view.columns:
    view = view[view["processor_brand"].isin(cpu_sel)]

# Show results
show_cols = [c for c in ["brand","model","price_myr","ram_gb","ssd",
                         "processor_brand","processor_gnrtn","p_fit"]
             if c in view.columns]
out = view.sort_values("p_fit", ascending=False)[show_cols].head(top_k)
st.subheader("Recommendations")
st.dataframe(out, use_container_width=True)

