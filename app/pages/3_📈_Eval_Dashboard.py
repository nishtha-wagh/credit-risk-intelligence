"""
app/pages/3_📈_Eval_Dashboard.py

Run batch evaluation on a holdout sample and display:
  - Metric summary cards (faithfulness, relevance, completeness, accuracy)
  - Score distribution histograms
  - Confusion matrix (predicted vs analyst tier)
  - Per-borrower results table
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.shared import (
    inject_css, load_borrowers, load_retriever, load_scorer,
    TIER_COLOR, data_missing_error,
)

st.set_page_config(page_title="Eval Dashboard", page_icon="📈", layout="wide")
inject_css()

st.title("📈 Evaluation Dashboard")
st.caption(
    "Run a batch evaluation on a held-out sample. "
    "Tracks faithfulness, relevance, completeness, and decision accuracy."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = load_borrowers()
if df is None:
    data_missing_error()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Eval settings")
    n_samples = st.slider("Sample size", 5, min(50, len(df)), 10)
    top_k     = st.slider("Chunks per assessment", 2, 10, 5)
    use_xgb   = st.checkbox("XGBoost + SHAP layer", value=False,
                             help="Slower but adds XGB accuracy metric")
    seed      = st.number_input("Random seed", value=99, step=1)
    run_eval  = st.button("▶  Run evaluation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Cached results storage (session state)
# ---------------------------------------------------------------------------
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
if run_eval:
    from evaluation.metrics import evaluate
    from generation.generator import generate_assessment

    retriever = load_retriever()
    holdout   = df.sample(n=n_samples, random_state=int(seed))

    scorer   = None
    explainer = None
    if use_xgb:
        try:
            scorer   = load_scorer()
            from generation.shap_explainer import SHAPExplainer
            explainer = SHAPExplainer(scorer)
        except Exception as e:
            st.warning(f"XGBoost unavailable: {e}")

    progress = st.progress(0, text="Running evaluations...")
    results  = []

    for idx, (_, row) in enumerate(holdout.iterrows()):
        bid          = row["borrower_id"]
        ground_truth = row.get("analyst_risk_tier")
        borrower_row = row.to_dict()

        filters = {"loan_type": borrower_row.get("loan_type")}
        chunks  = retriever.retrieve(
            query=f"credit risk signals for borrower {bid}",
            filters=filters, top_k=top_k,
        )

        xgb_signal  = scorer.score(borrower_row)  if scorer   else None
        shap_signal = explainer.explain(borrower_row) if explainer else None

        output     = generate_assessment(bid, borrower_row, chunks,
                                         xgb_signal=xgb_signal, shap_signal=shap_signal)
        eval_result = evaluate(output, chunks, ground_truth_tier=ground_truth)

        results.append({
            "borrower_id":        bid,
            "loan_type":          borrower_row.get("loan_type"),
            "ground_truth":       ground_truth,
            "predicted":          output.risk_tier,
            "confidence":         output.confidence,
            "retrieval_score":    output.retrieval_score,
            "sources_used":       output.sources_used,
            "latency_ms":         output.latency_ms,
            "faithfulness":       eval_result.faithfulness,
            "relevance":          eval_result.relevance,
            "completeness":       eval_result.completeness,
            "decision_accuracy":  eval_result.decision_accuracy or 0.0,
            "composite":          eval_result.composite,
            "xgb_tier":           output.xgb_predicted_tier,
            "xgb_prob":           output.xgb_probability,
        })

        progress.progress((idx + 1) / n_samples, text=f"Assessed {idx+1}/{n_samples}...")

    progress.empty()
    st.session_state.eval_results = pd.DataFrame(results)
    st.success(f"✅ Evaluation complete — {n_samples} borrowers assessed.")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
res = st.session_state.eval_results

if res is None:
    st.info("Configure settings in the sidebar and click **▶ Run evaluation**.")
    st.stop()

# Metric summary cards
st.subheader("Metric summary")
metrics = ["faithfulness", "relevance", "completeness", "decision_accuracy", "composite"]
labels  = ["Faithfulness", "Relevance", "Completeness", "Decision accuracy", "Composite"]
mc = st.columns(5)
for col, metric, label in zip(mc, metrics, labels):
    val  = res[metric].mean()
    col.metric(label, f"{val:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Score distribution histograms
# ---------------------------------------------------------------------------
st.subheader("Score distributions")

hist_cols = st.columns(4)
hist_metrics = [
    ("faithfulness",      "Faithfulness",      "#378ADD"),
    ("relevance",         "Relevance",         "#1D9E75"),
    ("completeness",      "Completeness",      "#EF9F27"),
    ("decision_accuracy", "Decision accuracy", "#D85A30"),
]

for col, (metric, label, color) in zip(hist_cols, hist_metrics):
    with col:
        fig, ax = plt.subplots(figsize=(3, 2.2))
        fig.patch.set_facecolor("#FAFAF8")
        ax.set_facecolor("#FAFAF8")
        ax.hist(res[metric], bins=10, range=(0, 1),
                color=color, alpha=0.80, edgecolor="white", linewidth=0.5)
        ax.axvline(res[metric].mean(), color="#2C2C2A", linewidth=1.2,
                   linestyle="--", label=f"Mean: {res[metric].mean():.2f}")
        ax.set_title(label, fontsize=10, fontweight="500")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, framealpha=0.8, edgecolor="#D3D1C7")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# Confusion matrix: predicted vs analyst tier
# ---------------------------------------------------------------------------
st.subheader("Predicted vs analyst tier")

tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
conf_left, conf_right = st.columns([1, 1])

with conf_left:
    # Build confusion matrix
    from sklearn.metrics import confusion_matrix
    gt_vals   = res["ground_truth"].values
    pred_vals = res["predicted"].values

    # Only include tiers that appear in the data
    present = [t for t in tier_order if t in gt_vals or t in pred_vals]
    cm = confusion_matrix(gt_vals, pred_vals, labels=present)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#FAFAF8")
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, fontsize=9)
    ax.set_yticklabels(present, fontsize=9)
    ax.set_xlabel("Predicted tier", fontsize=9)
    ax.set_ylabel("Analyst tier (ground truth)", fontsize=9)
    ax.set_title("Confusion matrix", fontsize=10, fontweight="500")

    for i in range(len(present)):
        for j in range(len(present)):
            val = cm[i, j]
            txt_color = "white" if val > cm.max() * 0.6 else "#2C2C2A"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, fontweight="600", color=txt_color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with conf_right:
    # Accuracy by tier
    st.markdown("**Accuracy by loan type**")
    loan_acc = (
        res.assign(correct=res["ground_truth"] == res["predicted"])
        .groupby("loan_type")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy", ascending=False)
    )
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    fig2.patch.set_facecolor("#FAFAF8")
    ax2.set_facecolor("#FAFAF8")
    colors = ["#1D9E75" if v >= 0.7 else "#EF9F27" if v >= 0.5 else "#E24B4A"
              for v in loan_acc["accuracy"]]
    bars = ax2.barh(loan_acc.index, loan_acc["accuracy"],
                    color=colors, alpha=0.85, height=0.5)
    for bar, (_, row_) in zip(bars, loan_acc.iterrows()):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.0%}  (n={int(row_['n'])})",
                 va="center", fontsize=8, color="#2C2C2A")
    ax2.set_xlim(0, 1.3)
    ax2.set_xlabel("Accuracy", fontsize=9)
    ax2.set_title("Decision accuracy by loan type", fontsize=10, fontweight="500")
    ax2.tick_params(labelsize=9)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="x", color="#D3D1C7", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

st.divider()

# ---------------------------------------------------------------------------
# Per-borrower results table
# ---------------------------------------------------------------------------
st.subheader("Per-borrower results")

display_cols = ["borrower_id", "loan_type", "ground_truth", "predicted",
                "confidence", "faithfulness", "relevance", "composite", "latency_ms"]
display_df = res[display_cols].copy()
display_df["confidence"]   = display_df["confidence"].map("{:.0%}".format)
display_df["faithfulness"] = display_df["faithfulness"].map("{:.2f}".format)
display_df["relevance"]    = display_df["relevance"].map("{:.2f}".format)
display_df["composite"]    = display_df["composite"].map("{:.3f}".format)
display_df["latency_ms"]   = display_df["latency_ms"].map("{:.0f}ms".format)

def _highlight_match(row):
    color = "#E1F5EE" if row["ground_truth"] == row["predicted"] else "#FAECE7"
    return [f"background-color: {color}"] * len(row)

st.dataframe(
    display_df.style.apply(_highlight_match, axis=1),
    use_container_width=True,
    height=350,
)

# Download button
csv = res.to_csv(index=False).encode()
st.download_button(
    "⬇  Download full results CSV",
    data=csv,
    file_name="eval_results.csv",
    mime="text/csv",
)
