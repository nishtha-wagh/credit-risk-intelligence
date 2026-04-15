"""
app/pages/4_🧩_Model_Explainability.py

Population-level XGBoost + SHAP explainability:
  - Feature importance bar chart (gain-based)
  - SHAP beeswarm summary plot across sampled borrowers
  - SHAP dependence plot for any feature pair
  - Score distribution by risk tier
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.shared import (
    inject_css, load_borrowers, load_scorer,
    TIER_COLOR, FEATURE_LABELS, data_missing_error,
)
from generation.xgb_scorer import FEATURES

st.set_page_config(page_title="Model Explainability", page_icon="🧩", layout="wide")
inject_css()

st.title("🧩 Model Explainability")
st.caption(
    "Population-level XGBoost feature importance and SHAP attributions. "
    "Understand what drives the model's risk predictions across the full borrower portfolio."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------
df = load_borrowers()
if df is None:
    data_missing_error()

try:
    scorer = load_scorer()
except Exception as e:
    st.error(f"XGBoost model unavailable: {e}\n\nRun: `python scripts/train_xgb.py`")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    n_shap   = st.slider("Borrowers for SHAP plots", 20, min(200, len(df)), 50)
    shap_seed = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.markdown("**Dependence plot**")
    feat_x = st.selectbox("Feature (X axis)", FEATURES,
                          index=FEATURES.index("dti_ratio") if "dti_ratio" in FEATURES else 0)
    feat_c = st.selectbox("Colour by", FEATURES,
                          index=FEATURES.index("fico_score") if "fico_score" in FEATURES else 1)

# ---------------------------------------------------------------------------
# Section 1: Feature importance
# ---------------------------------------------------------------------------
st.subheader("XGBoost feature importance (gain)")

importances = scorer._model.feature_importances_
imp_pairs   = sorted(zip(FEATURES, importances.tolist()), key=lambda x: x[1], reverse=True)

fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
fig_imp.patch.set_facecolor("#FAFAF8")
ax_imp.set_facecolor("#FAFAF8")

feat_labels = [FEATURE_LABELS.get(f, f) for f, _ in imp_pairs]
feat_vals   = [v for _, v in imp_pairs]
bar_colors  = [
    "#378ADD" if i == 0 else
    "#85B7EB" if i < 3  else
    "#B5D4F4"
    for i in range(len(feat_vals))
]

bars = ax_imp.barh(feat_labels[::-1], feat_vals[::-1],
                   color=bar_colors[::-1], alpha=0.90, height=0.6)
ax_imp.set_xlabel("Feature importance (gain)", fontsize=9)
ax_imp.tick_params(axis="y", labelsize=9)
ax_imp.tick_params(axis="x", labelsize=8)
ax_imp.spines[["top","right"]].set_visible(False)
ax_imp.grid(axis="x", color="#D3D1C7", linewidth=0.5)
for bar, val in zip(bars, feat_vals[::-1]):
    ax_imp.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="#2C2C2A")

plt.tight_layout()
st.pyplot(fig_imp, use_container_width=True)
plt.close(fig_imp)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: SHAP summary (beeswarm-style dot plot)
# ---------------------------------------------------------------------------
st.subheader("SHAP summary — feature impact across borrower population")
st.caption(f"Computed on a sample of {n_shap} borrowers")

@st.cache_data(show_spinner="Computing SHAP values...")
def compute_shap(n, seed):
    try:
        import shap
    except ImportError:
        return None, None, None

    sample = df.sample(n=n, random_state=int(seed))
    X = sample[FEATURES].fillna(sample[FEATURES].median(numeric_only=True))

    explainer  = shap.TreeExplainer(scorer._model)
    shap_vals  = explainer.shap_values(X)

    # Binary model: take class-1 (high risk) values
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    return sv, X, sample

shap_values, X_sample, sample_df = compute_shap(n_shap, int(shap_seed))

if shap_values is None:
    st.warning("SHAP not installed. Run: `pip install shap`")
else:
    # Beeswarm-style: one dot per borrower per feature, colour = feature value
    shap_df  = pd.DataFrame(shap_values, columns=FEATURES)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.index.tolist()[:10]

    fig_bee, ax_bee = plt.subplots(figsize=(9, 5))
    fig_bee.patch.set_facecolor("#FAFAF8")
    ax_bee.set_facecolor("#FAFAF8")

    cmap   = plt.get_cmap("RdYlGn_r")
    yticks = []
    ylabels = []

    for i, feat in enumerate(reversed(top_feats)):
        sv_col    = shap_df[feat].values
        feat_vals = X_sample[feat].values
        # Normalise feature values to [0,1] for colouring
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax > fmin:
            norm_vals = (feat_vals - fmin) / (fmax - fmin)
        else:
            norm_vals = np.full_like(feat_vals, 0.5)

        # Jitter y
        jitter = np.random.default_rng(i).uniform(-0.2, 0.2, size=len(sv_col))
        y_pos  = np.full(len(sv_col), i)

        ax_bee.scatter(
            sv_col, y_pos + jitter,
            c=[cmap(v) for v in norm_vals],
            alpha=0.6, s=14, edgecolors="none", zorder=3,
        )
        yticks.append(i)
        ylabels.append(FEATURE_LABELS.get(feat, feat))

    ax_bee.axvline(0, color="#888780", linewidth=0.8, zorder=4)
    ax_bee.set_yticks(yticks)
    ax_bee.set_yticklabels(ylabels, fontsize=9)
    ax_bee.set_xlabel("SHAP value (impact on high-risk probability)", fontsize=9)
    ax_bee.tick_params(axis="x", labelsize=8)
    ax_bee.spines[["top","right","left"]].set_visible(False)
    ax_bee.grid(axis="x", color="#D3D1C7", linewidth=0.5, zorder=0)

    # Colourbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_bee, shrink=0.5, pad=0.01)
    cbar.set_label("Feature value\n(low → high)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    st.pyplot(fig_bee, use_container_width=True)
    plt.close(fig_bee)

    st.divider()

    # ---------------------------------------------------------------------------
    # Section 3: SHAP dependence plot
    # ---------------------------------------------------------------------------
    st.subheader(f"SHAP dependence — {FEATURE_LABELS.get(feat_x, feat_x)}")
    st.caption("How a single feature's SHAP value changes with its raw value, coloured by a second feature")

    sv_x    = shap_df[feat_x].values
    raw_x   = X_sample[feat_x].values
    raw_c   = X_sample[feat_c].values
    cmin, cmax = raw_c.min(), raw_c.max()
    if cmax > cmin:
        norm_c = (raw_c - cmin) / (cmax - cmin)
    else:
        norm_c = np.full_like(raw_c, 0.5)

    fig_dep, ax_dep = plt.subplots(figsize=(7, 3.5))
    fig_dep.patch.set_facecolor("#FAFAF8")
    ax_dep.set_facecolor("#FAFAF8")

    sc = ax_dep.scatter(raw_x, sv_x, c=[cmap(v) for v in norm_c],
                        alpha=0.7, s=22, edgecolors="none", zorder=3)
    ax_dep.axhline(0, color="#888780", linewidth=0.8, linestyle="--", zorder=4)
    ax_dep.set_xlabel(FEATURE_LABELS.get(feat_x, feat_x), fontsize=9)
    ax_dep.set_ylabel(f"SHAP({FEATURE_LABELS.get(feat_x, feat_x)})", fontsize=9)
    ax_dep.tick_params(labelsize=8)
    ax_dep.spines[["top","right"]].set_visible(False)
    ax_dep.grid(color="#D3D1C7", linewidth=0.5, zorder=0)

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(cmin, cmax))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax_dep, shrink=0.8, pad=0.01)
    cbar2.set_label(FEATURE_LABELS.get(feat_c, feat_c), fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    plt.tight_layout()
    st.pyplot(fig_dep, use_container_width=True)
    plt.close(fig_dep)

    st.divider()

# ---------------------------------------------------------------------------
# Section 4: Score distribution by risk tier
# ---------------------------------------------------------------------------
st.subheader("XGBoost score distribution by analyst risk tier")

@st.cache_data(show_spinner="Scoring full dataset...")
def score_all():
    scores = []
    for _, row in df.iterrows():
        try:
            sig = scorer.score(row.to_dict())
            scores.append({
                "borrower_id": row["borrower_id"],
                "tier": row["analyst_risk_tier"],
                "xgb_prob": sig.probability,
            })
        except Exception:
            pass
    return pd.DataFrame(scores)

scored_df = score_all()

tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
fig_dist, ax_dist = plt.subplots(figsize=(8, 3.5))
fig_dist.patch.set_facecolor("#FAFAF8")
ax_dist.set_facecolor("#FAFAF8")

for tier in tier_order:
    sub = scored_df[scored_df["tier"] == tier]["xgb_prob"]
    if len(sub) == 0:
        continue
    ax_dist.hist(sub, bins=20, range=(0, 1), alpha=0.65,
                 label=f"{tier} (n={len(sub)})",
                 color=TIER_COLOR.get(tier, "#888"),
                 edgecolor="white", linewidth=0.4)

ax_dist.set_xlabel("XGBoost P(HIGH + CRITICAL)", fontsize=9)
ax_dist.set_ylabel("Count", fontsize=9)
ax_dist.set_xlim(0, 1)
ax_dist.tick_params(labelsize=8)
ax_dist.spines[["top","right"]].set_visible(False)
ax_dist.grid(axis="y", color="#D3D1C7", linewidth=0.5)
ax_dist.legend(fontsize=8, framealpha=0.85, edgecolor="#D3D1C7")
plt.tight_layout()
st.pyplot(fig_dist, use_container_width=True)
plt.close(fig_dist)
