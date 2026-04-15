"""
generation/context_builder.py

Assembles the full context block passed to the LLM.

Layers (each optional, gracefully degraded if unavailable):
  1. Structured signals       — raw borrower fields, formatted as key: value
  2. XGBoost risk score       — model probability + predicted tier
  3. SHAP feature attributions — top N features driving the XGBoost score
  4. Retrieved case notes     — ranked unstructured chunks from hybrid retrieval

Design principle: the LLM receives a richer, machine-augmented context than
raw data alone. XGBoost adds a calibrated probability; SHAP makes the model's
reasoning transparent so the LLM can reference it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.hybrid_retriever import RetrievedChunk

# Fields excluded from the structured signals block (noise / not useful to LLM)
_EXCLUDE_FIELDS = {
    "note_text", "record_updated_at", "origination_date",
    "last_payment_date", "loan_id",
}

# Human-readable labels for structured fields
_FIELD_LABELS: dict[str, str] = {
    "borrower_id":           "Borrower ID",
    "loan_type":             "Loan type",
    "vintage_year":          "Vintage year",
    "loan_amount":           "Loan amount ($)",
    "loan_term_months":      "Loan term (months)",
    "fico_score":            "FICO score",
    "dti_ratio":             "Debt-to-income ratio",
    "ltv_ratio":             "Loan-to-value ratio",
    "num_open_accounts":     "Open accounts",
    "credit_history_yrs":    "Credit history (years)",
    "payments_on_time":      "On-time payments",
    "payments_late_30d":     "Late payments (30d)",
    "payments_late_60d":     "Late payments (60d)",
    "payments_late_90d":     "Late payments (90d)",
    "num_deferrals":         "Deferrals",
    "ever_defaulted":        "Prior default",
    "employment_status":     "Employment status",
    "employment_gap_months": "Employment gap (months)",
    "annual_income":         "Annual income ($)",
    "in_collections":        "In collections",
    "charged_off":           "Charged off",
    "analyst_risk_tier":     "Analyst risk tier (label)",
}


@dataclass
class XGBoostSignal:
    """Output from the XGBoost credit scorer."""
    predicted_tier: str          # LOW | MEDIUM | HIGH | CRITICAL
    probability: float           # P(HIGH or CRITICAL) — 0.0–1.0
    class_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class SHAPSignal:
    """Top SHAP feature attributions for the XGBoost prediction."""
    top_features: list[tuple[str, float]]  # [(feature_name, shap_value), ...]
    base_value: float = 0.0


def build_context(
    borrower_row: dict,
    retrieved_chunks: list[RetrievedChunk],
    xgb_signal: XGBoostSignal | None = None,
    shap_signal: SHAPSignal | None = None,
) -> str:
    """
    Assemble the full context string for the LLM prompt.

    Sections are clearly delimited so the LLM can reference each by name.
    Absent optional sections are omitted entirely (no empty headers).
    """
    sections: list[str] = []

    sections.append(_format_structured(borrower_row))

    if xgb_signal is not None:
        sections.append(_format_xgboost(xgb_signal))

    if shap_signal is not None:
        sections.append(_format_shap(shap_signal))

    if retrieved_chunks:
        sections.append(_format_chunks(retrieved_chunks))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------

def _format_structured(row: dict) -> str:
    lines = ["=== STRUCTURED BORROWER SIGNALS ==="]
    for key, value in row.items():
        if key in _EXCLUDE_FIELDS:
            continue
        label = _FIELD_LABELS.get(key, key.replace("_", " ").title())
        formatted = _fmt_value(key, value)
        lines.append(f"  {label}: {formatted}")
    return "\n".join(lines)


def _format_xgboost(sig: XGBoostSignal) -> str:
    lines = ["=== XGBOOST MODEL SCORE ==="]
    lines.append(f"  Predicted tier   : {sig.predicted_tier}")
    lines.append(f"  Default probability (HIGH+CRITICAL): {sig.probability:.1%}")
    if sig.class_probabilities:
        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            p = sig.class_probabilities.get(tier, 0.0)
            bar = "█" * int(p * 20)
            lines.append(f"  P({tier:<8s})    : {p:.1%}  {bar}")
    return "\n".join(lines)


def _format_shap(sig: SHAPSignal) -> str:
    lines = ["=== SHAP FEATURE ATTRIBUTIONS (top drivers) ==="]
    lines.append(f"  Base value (avg prediction): {sig.base_value:.3f}")
    lines.append("  Feature contributions toward HIGH/CRITICAL risk:")
    for feature, value in sig.top_features:
        label = _FIELD_LABELS.get(feature, feature.replace("_", " ").title())
        direction = "▲ increases risk" if value > 0 else "▼ decreases risk"
        lines.append(f"  {label:<35s}  {value:+.4f}  ({direction})")
    return "\n".join(lines)


def _format_chunks(chunks: list) -> str:
    lines = ["=== RETRIEVED CASE NOTES ==="]
    for i, chunk in enumerate(chunks, 1):
        header = f"  [{i}] {chunk.note_type.upper()} | similarity={chunk.similarity_score:.3f}"
        lines.append(header)
        # Indent chunk text for visual separation
        for text_line in chunk.text.strip().splitlines():
            lines.append(f"      {text_line}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Value formatter
# ---------------------------------------------------------------------------

def _fmt_value(key: str, value) -> str:
    if value is None or value != value:   # None or NaN
        return "N/A"
    if key == "dti_ratio" or key == "ltv_ratio":
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return str(value)
    if key in {"loan_amount", "annual_income"}:
        try:
            return f"${float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
