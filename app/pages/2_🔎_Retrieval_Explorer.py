"""
app/pages/2_🔎_Retrieval_Explorer.py

Makes the hybrid retrieval layer fully transparent:
  - Enter a free-text query
  - Choose metadata filters
  - See exactly which chunks were returned, their scores, and why
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.shared import (
    inject_css, load_borrowers, load_case_notes, load_retriever,
    data_missing_error,
)

st.set_page_config(page_title="Retrieval Explorer", page_icon="🔎", layout="wide")
inject_css()

st.title("🔎 Retrieval Explorer")
st.caption(
    "Inspect the hybrid retrieval layer. See how metadata pre-filtering "
    "shapes the candidate pool before vector similarity ranking."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df    = load_borrowers()
notes = load_case_notes()
if df is None or notes is None:
    data_missing_error()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Query settings")

    query = st.text_area(
        "Search query",
        value="borrower with multiple deferrals and employment gap",
        height=80,
    )
    top_k     = st.slider("Top-K results", 3, 20, 8)
    min_score = st.slider("Min similarity score", 0.0, 1.0, 0.0, step=0.05)

    st.markdown("---")
    st.markdown("**Metadata filters** (optional)")
    loan_types  = ["(none)"] + sorted(df["loan_type"].unique().tolist())
    note_types  = ["(none)", "underwriter", "collections", "servicing", "complaint"]
    risk_bands  = ["(none)", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    vintage_yrs = ["(none)"] + sorted(df["vintage_year"].dropna().astype(int).unique().tolist(), reverse=True)

    sel_loan    = st.selectbox("Loan type",    loan_types)
    sel_note    = st.selectbox("Note type",    note_types)
    sel_risk    = st.selectbox("Risk band",    risk_bands)
    sel_vintage = st.selectbox("Vintage year", vintage_yrs)

    run = st.button("▶  Search", type="primary", use_container_width=True)

# Build filters dict
filters = {}
if sel_loan    != "(none)": filters["loan_type"]    = sel_loan
if sel_note    != "(none)": filters["note_type"]     = sel_note
if sel_risk    != "(none)": filters["risk_band"]     = sel_risk
if sel_vintage != "(none)": filters["vintage_year"]  = int(sel_vintage)

# ---------------------------------------------------------------------------
# Filter visualisation — always show
# ---------------------------------------------------------------------------
col_info, col_pool = st.columns([1, 1])

with col_info:
    st.markdown("**Active filters**")
    if filters:
        for k, v in filters.items():
            st.markdown(
                f'<div style="display:inline-block;background:#E6F1FB;color:#185FA5;'
                f'border-radius:99px;padding:3px 12px;font-size:12px;margin:3px 4px 3px 0;">'
                f'{k}: <strong>{v}</strong></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span style="color:#5F5E5A;font-size:13px;">None — searching all chunks</span>',
            unsafe_allow_html=True,
        )

with col_pool:
    # Show how many notes/chunks survive the filter
    filtered_notes = notes.copy()
    if sel_loan    != "(none)": filtered_notes = filtered_notes[filtered_notes["loan_type"]  == sel_loan]
    if sel_note    != "(none)": filtered_notes = filtered_notes[filtered_notes["note_type"]  == sel_note]
    if sel_risk    != "(none)": filtered_notes = filtered_notes[filtered_notes["risk_band"]  == sel_risk]
    if sel_vintage != "(none)": filtered_notes = filtered_notes[filtered_notes["vintage_year"] == int(sel_vintage)]

    total = len(notes)
    kept  = len(filtered_notes)
    pct   = kept / total * 100 if total else 0

    st.markdown(
        f'<div style="border-radius:8px;padding:10px 16px;background:#E1F5EE;'
        f'border:1px solid #5DCAA5;font-size:13px;">'
        f'Metadata filter passes <strong>{kept}</strong> of <strong>{total}</strong> '
        f'notes into vector search &nbsp;({pct:.0f}%)</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Run retrieval
# ---------------------------------------------------------------------------
if not run:
    st.info("Set your query and filters, then click **▶ Search**.")
    st.stop()

if not query.strip():
    st.warning("Please enter a search query.")
    st.stop()

retriever = load_retriever()

with st.spinner("Running hybrid retrieval..."):
    chunks = retriever.retrieve(
        query=query,
        filters=filters if filters else None,
        top_k=top_k,
        min_score=min_score,
    )

st.subheader(f"Results — {len(chunks)} chunk(s) returned")

if not chunks:
    st.warning("No chunks returned. Try lowering min similarity score or removing filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Similarity score bar chart
# ---------------------------------------------------------------------------
scores   = [c.similarity_score for c in chunks]
labels   = [f"#{i+1} {c.note_type[:4].upper()} {c.borrower_id}" for i, c in enumerate(chunks)]
bar_cols = ["#378ADD" if s >= 0.80 else "#888780" for s in scores]

fig, ax = plt.subplots(figsize=(8, max(2.5, len(chunks) * 0.38)))
fig.patch.set_facecolor("#FAFAF8")
ax.set_facecolor("#FAFAF8")
bars = ax.barh(range(len(labels)), scores, color=bar_cols, alpha=0.85, height=0.55, zorder=3)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Cosine similarity score", fontsize=9)
ax.axvline(min_score, color="#E24B4A", linewidth=1, linestyle="--", zorder=4, label=f"Min score ({min_score:.2f})")
ax.grid(axis="x", color="#D3D1C7", linewidth=0.5, zorder=0)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="x", labelsize=8)
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{score:.3f}", va="center", fontsize=8, color="#2C2C2A")
ax.legend(fontsize=8, loc="lower right", framealpha=0.8, edgecolor="#D3D1C7")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# Chunk cards
# ---------------------------------------------------------------------------
NOTE_COLORS = {
    "underwriter": "#185FA5",
    "collections": "#993C1D",
    "servicing":   "#0F6E56",
    "complaint":   "#993556",
}

for i, chunk in enumerate(chunks, 1):
    nc = NOTE_COLORS.get(chunk.note_type, "#5F5E5A")

    with st.expander(
        f"#{i}  {chunk.note_type.upper()}  ·  {chunk.borrower_id}  ·  score: {chunk.similarity_score:.3f}",
        expanded=(i <= 3),
    ):
        meta_col, text_col = st.columns([1, 2])

        with meta_col:
            st.markdown("**Chunk metadata**")
            meta = chunk.metadata
            rows = [
                ("Borrower",   chunk.borrower_id),
                ("Note type",  chunk.note_type),
                ("Loan type",  meta.get("loan_type", "—")),
                ("Vintage",    meta.get("vintage_year", "—")),
                ("Risk band",  meta.get("risk_band", "—")),
                ("Chunk idx",  meta.get("chunk_index", "—")),
                ("Tokens",     meta.get("token_estimate", "—")),
            ]
            for label, val in rows:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;border-bottom:0.5px solid #eee;font-size:12px;">'
                    f'<span style="color:#5F5E5A">{label}</span>'
                    f'<span style="font-weight:500">{val}</span></div>',
                    unsafe_allow_html=True,
                )

        with text_col:
            st.markdown("**Chunk text**")
            st.markdown(
                f'<div style="border-left:3px solid {nc};padding:10px 14px;'
                f'background:{nc}11;border-radius:0 8px 8px 0;'
                f'font-size:13px;line-height:1.7;">{chunk.text}</div>',
                unsafe_allow_html=True,
            )
