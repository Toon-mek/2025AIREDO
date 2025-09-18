# app.py — Logistic Regression, simple but better
import os, numpy as np, pandas as pd, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Laptop Recommender — Logistic", layout="wide")
st.title("Laptop Recommender — Logistic Regression")

# ------------------ Data ------------------
@st.cache_data
def load_data():
    for p in ["data/logistics_regression.csv", "logistics_regression.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("CSV not found in 'data/' or repo root.")

df_raw = load_data()

# ------------------ Sidebar controls ------------------
with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Budget (MYR)", 1000, 20000, 3000, 100)
    K      = st.slider("Top-K", 1, 30, 10)
    strict = st.checkbox("Strict (drop rows with any missing key fields)", True)

    st.divider()
    st.subheader("Label rule (fit=1)")
    ram_min = st.select_slider("RAM ≥ (GB)", options=[4,8,12,16,24,32], value=8)
    ssd_min = st.select_slider("SSD ≥ (GB)", options=[128,256,512,1024], value=256)

    st.subheader("Model")
    balanced = st.checkbox("class_weight='balanced'", True)
    C = st.select_slider("Regularization (C)", options=[0.1,0.5,1.0,2.0,5.0,10.0], value=1.0)

    st.subheader("Threshold")
    thr_mode = st.radio("Decision threshold", ["0.50", "Best F1 (auto)"], index=0)

# ------------------ Labels + split ------------------
df = df_raw.copy()
rg = pd.to_numeric(df.get("ram_gb"), errors="coerce")
sd = pd.to_numeric(df.get("ssd"),    errors="coerce")
df["fit_cs"] = ((rg >= ram_min) & (sd >= ssd_min)).astype(int)
if df["fit_cs"].sum() == 0:
    st.warning("No positives under current rule; try relaxing thresholds.")
X = df.drop(columns=["price_myr","fit_cs"], errors="ignore")
y = df["fit_cs"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y if y.nunique()==2 else None
)

# ------------------ Preprocess + Logistic ------------------
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

clf = Pipeline([
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=1000, class_weight=("balanced" if balanced else None), C=C, random_state=42)),
]).fit(X_train, y_train)

# ------------------ Metrics ------------------
proba = clf.predict_proba(X_test)[:,1]

# choose threshold
if thr_mode == "Best F1 (auto)":
    ths = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y_test, (proba>=t).astype(int), zero_division=0) for t in ths]
    thr = float(ths[int(np.argmax(f1s))])
else:
    thr = 0.50

pred = (proba >= thr).astype(int)
P = precision_score(y_test, pred, zero_division=0)
R = recall_score(y_test, pred, zero_division=0)
F = f1_score(y_test, pred, zero_division=0)

# Precision@K
order = np.argsort(-proba)
def prec_at_k(k:int)->float:
    if k<1: return np.nan
    topk = y_test.iloc[order[:min(k, len(order))]]
    return float(topk.mean()) if len(topk) else np.nan

colA, colB = st.columns(2)
with colA:
    st.subheader("Overall Metrics")
    st.write(f"**Precision**: {P:.3f}  |  **Recall**: {R:.3f}  |  **F1**: {F:.3f}  |  **Thr**: {thr:.2f}")
    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    fig = plt.figure()
    plt.imshow(cm)
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v}", ha="center", va="center")
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    plt.title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)

with colB:
    st.subheader("Precision–Recall")
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, proba)
    fig2 = plt.figure()
    plt.plot(pr_rec, pr_prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    st.pyplot(fig2, clear_figure=True)
    st.write(f"**Precision@{K}**: {prec_at_k(K):.3f}")

# ------------------ Recommendations (Top-K under budget) ------------------
X_all = df.drop(columns=["price_myr","fit_cs"], errors="ignore")
df["p_fit"] = clf.predict_proba(X_all)[:,1]

show = [c for c in ["brand","model","price_myr","ram_gb","ssd","processor_brand","processor_gnrtn","p_fit"] if c in df.columns]
view = df.query("price_myr <= @budget")[show]
if strict:
    view = view.dropna(how="any")

st.subheader("Recommendations")
out = view.sort_values("p_fit", ascending=False).head(K)
st.dataframe(out, use_container_width=True)
st.download_button("Download Top-K CSV", out.to_csv(index=False).encode(), file_name="topk.csv")

# ------------------ Top features (weights) ------------------
try:
    pre = clf.named_steps["pre"]
    oh = pre.named_transformers_["cat"].named_steps["oh"]
    num = pre.transformers_[0][2]
    cat = pre.transformers_[1][2]
    feat_names = list(num) + list(oh.get_feature_names_out(cat))
    weights = clf.named_steps["lr"].coef_[0]
    imp = pd.DataFrame({"feature": feat_names, "weight": weights}).sort_values("weight", ascending=False)
    st.subheader("Top features")
    st.write("Increase p_fit (top +):")
    st.dataframe(imp.head(10))
    st.write("Decrease p_fit (top −):")
    st.dataframe(imp.tail(10))
except Exception:
    pass
