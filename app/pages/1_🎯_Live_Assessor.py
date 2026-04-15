"""
app/pages/1_🎯_Live_Assessor.py

Full RAG pipeline demo:
  - Select borrower → run hybrid retrieval → XGBoost score → SHAP → LLM generation
  - Shows decision, key signals, SHAP waterfall chart, retrieved evidence, raw JSON
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.shared import (
    inject_css, load_borrowers, load_retriever, load_scorer, load_explainer,
    TIER_COLOR, TIER_BG, TIER_ICON, FEATURE_LABELS, tier_badge, data_missing_error,
)

st.set_page_config(page_title="Live Assessor", page_icon="🎯", layout="wide")
inject_css()

st.title("🎯 Live Assessor")
st.caption("Run the full pipeline end-to-end: retrieval → XGBoost → SHAP → LLM → structured decision")
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
    st.header("Controls")

    # Borrower filter
    loan_type_filter = st.selectbox("Filter by loan type", ["All"] + sorted(df["loan_type"].unique().tolist()))
    tier_filter      = st.selectbox("Filter by risk tier", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])

    filtered = df.copy()
    if loan_type_filter != "All":
        filtered = filtered[filtered["loan_type"] == loan_type_filter]
    if tier_filter != "All":
        filtered = filtered[filtered["analyst_risk_tier"] == tier_filter]

    if filtered.empty:
        st.warning("No borrowers match filters.")
        st.stop()

    selected_id  = st.selectbox("Borrower ID", filtered["borrower_id"].tolist())
    top_k        = st.slider("Chunks to retrieve", 2, 10, 5)
    use_filter   = st.checkbox("Metadata filter (loan type)", value=True)
    use_xgb      = st.checkbox("XGBoost + SHAP layer", value=True)

    run = st.button("▶  Run assessment", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Borrower snapshot (always visible)
# ---------------------------------------------------------------------------
borrower_row = df[df["borrower_id"] == selected_id].iloc[0].to_dict()

st.subheader(f"Borrower {selected_id}")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
analyst_tier = borrower_row.get("analyst_risk_tier", "")
c1.metric("Loan type",     borrower_row.get("loan_type", "").upper())
c2.metric("FICO score",    borrower_row.get("fico_score"))
c3.metric("DTI ratio",     f"{borrower_row.get('dti_ratio', 0):.1%}")
c4.metric("Late 30d",      borrower_row.get("payments_late_30d"))
c5.metric("Deferrals",     borrower_row.get("num_deferrals"))
c6.metric("Empl. gap (m)", borrower_row.get("employment_gap_months"))
c7.markdown(
    f"<div style='padding-top:28px'>{tier_badge(analyst_tier)}</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if not run:
    st.info("Configure options in the sidebar and click **▶ Run assessment**.")
    st.stop()

retriever = load_retriever()
filters   = {"loan_type": borrower_row["loan_type"]} if use_filter else None

# XGBoost + SHAP
xgb_signal  = None
shap_signal = None
if use_xgb:
    try:
        scorer     = load_scorer()
        explainer  = load_explainer(scorer)
        xgb_signal  = scorer.score(borrower_row)
        shap_signal = explainer.explain(borrower_row, top_n=10)
    except Exception as e:
        st.warning(f"XGBoost/SHAP unavailable: {e}")

# Retrieval + generation
with st.spinner("Retrieving context and generating assessment..."):
    from generation.generator import generate_assessment
    chunks = retriever.retrieve(
        query=f"credit risk assessment borrower {selected_id}",
        filters=filters,
        top_k=top_k,
    )
    result = generate_assessment(
        selected_id, borrower_row, chunks,
        xgb_signal=xgb_signal,
        shap_signal=shap_signal,
    )

# ---------------------------------------------------------------------------
# Output — two columns
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

# LEFT: Decision output
with left:
    tier   = result.risk_tier
    color  = TIER_COLOR.get(tier, "#888")
    bg     = TIER_BG.get(tier, "#eee")
    icon   = TIER_ICON.get(tier, "⚪")

    st.markdown(
        f'<div style="border-radius:12px;padding:20px 24px;background:{bg};'
        f'border:1.5px solid {color}55;margin-bottom:16px;">'
        f'<div style="font-size:13px;color:{color};font-weight:600;letter-spacing:.06em;'
        f'text-transform:uppercase;margin-bottom:4px">Risk assessment</div>'
        f'<div style="font-size:32px;font-weight:700;color:{color}">{icon} {tier}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence",    f"{result.confidence:.0%}")
    m2.metric("Retrieval score", f"{result.retrieval_score:.2f}")
    m3.metric("Latency",       f"{result.latency_ms}ms")

    if xgb_signal:
        st.markdown("---")
        x1, x2 = st.columns(2)
        x1.metric("XGB tier",          xgb_signal.predicted_tier)
        x2.metric("P(HIGH+CRITICAL)",  f"{xgb_signal.probability:.1%}")

    st.markdown("---")
    st.markdown(
        f'<div style="background:#F1EFE8;border-radius:8px;padding:12px 16px;'
        f'font-size:13px;line-height:1.6;margin-bottom:12px;">'
        f'<strong>Decision:</strong> {result.decision}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Key signals**")
    for sig in result.key_signals:
        st.markdown(
            f'<div class="signal-pill">◆ {sig}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Full reasoning"):
        st.write(result.reasoning)

    with st.expander("Raw JSON"):
        st.json(result.to_dict())

# RIGHT: SHAP waterfall + retrieved chunks
with right:
    # SHAP waterfall chart
    if shap_signal and shap_signal.top_features:
        st.markdown("**SHAP feature attributions** — what drove this score")

        features = [FEATURE_LABELS.get(f, f) for f, _ in shap_signal.top_features]
        values   = [v for _, v in shap_signal.top_features]
        colors   = [TIER_COLOR["HIGH"] if v > 0 else TIER_COLOR["LOW"] for v in values]

        fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.45)))
        fig.patch.set_facecolor("#FAFAF8")
        ax.set_facecolor("#FAFAF8")

        bars = ax.barh(
            range(len(features)), values,
            color=colors, alpha=0.85, height=0.6, zorder=3,
        )

        ax.axvline(0, color="#888780", linewidth=0.8, zorder=4)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("SHAP value (impact on risk probability)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#D3D1C7", linewidth=0.5, zorder=0)
        ax.spines[["top","right","left"]].set_visible(False)

        # Value labels
        for bar, val in zip(bars, values):
            x_pos = bar.get_width()
            ax.text(
                x_pos + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                color="#2C2C2A",
            )

        red_patch   = mpatches.Patch(color=TIER_COLOR["HIGH"],  alpha=0.85, label="Increases risk")
        green_patch = mpatches.Patch(color=TIER_COLOR["LOW"],   alpha=0.85, label="Decreases risk")
        ax.legend(handles=[red_patch, green_patch], fontsize=8, loc="lower right",
                  framealpha=0.8, edgecolor="#D3D1C7")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.info("Enable **XGBoost + SHAP layer** in the sidebar to see feature attributions.")

    st.markdown("---")

    # Retrieved chunks
    st.markdown(f"**Retrieved evidence** — {len(chunks)} chunk(s) used")
    if chunks:
        for i, chunk in enumerate(chunks, 1):
            note_color = {
                "underwriter": "#185FA5",
                "collections": "#993C1D",
                "servicing":   "#0F6E56",
                "complaint":   "#993556",
            }.get(chunk.note_type, "#5F5E5A")

            st.markdown(
                f'<div class="chunk-card" style="border-left-color:{note_color};">'
                f'<div style="font-size:11px;color:{note_color};font-weight:600;'
                f'margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em;">'
                f'{chunk.note_type} &nbsp;·&nbsp; score: {chunk.similarity_score:.3f}</div>'
                f'{chunk.text[:280]}{"…" if len(chunk.text) > 280 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No chunks retrieved — try lowering the minimum similarity score.")
