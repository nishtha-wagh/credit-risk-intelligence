"""
app/utils/shared.py

Shared constants, cached loaders, and styling helpers
used across all pages of the Streamlit demo.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make project root importable from any page
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_COLOR = {
    "LOW":      "#1D9E75",
    "MEDIUM":   "#EF9F27",
    "HIGH":     "#D85A30",
    "CRITICAL": "#E24B4A",
}

TIER_BG = {
    "LOW":      "#E1F5EE",
    "MEDIUM":   "#FAEEDA",
    "HIGH":     "#FAECE7",
    "CRITICAL": "#FCEBEB",
}

TIER_ICON = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🟠",
    "CRITICAL": "🔴",
}

FEATURE_LABELS = {
    "fico_score":            "FICO Score",
    "dti_ratio":             "Debt-to-Income Ratio",
    "ltv_ratio":             "Loan-to-Value Ratio",
    "payments_late_30d":     "Late Payments (30d)",
    "payments_late_60d":     "Late Payments (60d)",
    "payments_late_90d":     "Late Payments (90d)",
    "num_deferrals":         "Number of Deferrals",
    "employment_gap_months": "Employment Gap (months)",
    "num_open_accounts":     "Open Accounts",
    "credit_history_yrs":    "Credit History (years)",
    "annual_income":         "Annual Income",
    "loan_amount":           "Loan Amount",
    "loan_term_months":      "Loan Term (months)",
}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_borrowers() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "borrowers.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_case_notes() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "case_notes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_retriever():
    from retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever()


@st.cache_resource
def load_scorer():
    from generation.xgb_scorer import CreditScorer
    scorer = CreditScorer()
    model_path = ROOT / "data" / "processed" / "xgb_model.json"
    if model_path.exists():
        scorer.load(model_path)
    else:
        df = load_borrowers()
        if df is not None:
            scorer.train(df, save=True)
    return scorer


@st.cache_resource
def load_explainer(_scorer):
    from generation.shap_explainer import SHAPExplainer
    return SHAPExplainer(_scorer)


# ---------------------------------------------------------------------------
# Shared CSS injected once per page
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown("""
    <style>
    /* Sidebar nav polish */
    [data-testid="stSidebarNav"] { padding-top: 1rem; }

    /* Metric value sizing */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }

    /* Risk tier badge */
    .tier-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 99px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Signal pill */
    .signal-pill {
        display: inline-block;
        background: #F1EFE8;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 13px;
        line-height: 1.5;
        width: 100%;
    }

    /* Chunk card */
    .chunk-card {
        border-left: 3px solid #378ADD;
        padding: 8px 14px;
        margin: 8px 0;
        background: #E6F1FB22;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Divider override */
    hr { margin: 1rem 0 !important; }
    </style>
    """, unsafe_allow_html=True)


def tier_badge(tier: str) -> str:
    color = TIER_COLOR.get(tier, "#888")
    bg    = TIER_BG.get(tier, "#eee")
    return f'<span class="tier-badge" style="color:{color};background:{bg};">{tier}</span>'


def data_missing_error():
    st.error(
        "**No data found.** Run the setup scripts first:\n\n"
        "```bash\n"
        "python scripts/generate_mock_data.py\n"
        "python scripts/build_index.py\n"
        "python scripts/train_xgb.py\n"
        "```"
    )
    st.stop()
