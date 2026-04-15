"""
generation/shap_explainer.py

Computes SHAP (SHapley Additive exPlanations) feature attributions
for a single borrower row using the trained XGBoost model.

Returns a SHAPSignal consumed by context_builder.build_context().

Why this matters for the portfolio:
  The LLM receives not just raw feature values, but the *direction and
  magnitude* of each feature's contribution to the risk score. This allows
  the LLM to say "SHAP indicates DTI is the primary driver" rather than
  just "DTI is high" — producing more precise, auditable reasoning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generation.context_builder import SHAPSignal
from generation.xgb_scorer import FEATURES, CreditScorer


class SHAPExplainer:
    """
    Wraps a SHAP TreeExplainer for the CreditScorer model.

    Usage:
        explainer = SHAPExplainer(scorer)
        signal = explainer.explain(borrower_row, top_n=6)
    """

    def __init__(self, scorer: CreditScorer):
        if not scorer._is_trained:
            raise RuntimeError("CreditScorer must be trained before building SHAPExplainer.")
        try:
            import shap
        except ImportError:
            raise ImportError("Run: pip install shap")

        import shap as shap_lib
        self._explainer = shap_lib.TreeExplainer(scorer._model)
        self._scorer = scorer

    def explain(self, borrower_row: dict, top_n: int = 6) -> SHAPSignal:
        """
        Compute SHAP values for a single borrower.

        Returns SHAPSignal with the top_n features ranked by |SHAP value|.
        Positive SHAP → increases predicted risk probability.
        Negative SHAP → decreases predicted risk probability.
        """
        row_df = self._row_to_df(borrower_row)
        shap_values = self._explainer.shap_values(row_df)

        # For binary XGBClassifier, shap_values is shape (1, n_features)
        # We take the values for class=1 (high risk)
        if isinstance(shap_values, list):
            # Older SHAP versions return list[array] for multi-output
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        base_value = float(self._explainer.expected_value)
        if isinstance(self._explainer.expected_value, (list, np.ndarray)):
            base_value = float(self._explainer.expected_value[1])

        # Pair features with SHAP values, sort by absolute magnitude
        pairs = list(zip(FEATURES, values.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = pairs[:top_n]

        return SHAPSignal(
            top_features=[(f, round(v, 5)) for f, v in top_features],
            base_value=round(base_value, 4),
        )

    def _row_to_df(self, row: dict) -> pd.DataFrame:
        data = {f: [row.get(f, np.nan)] for f in FEATURES}
        df = pd.DataFrame(data)
        df = df.fillna(df.median(numeric_only=True))
        return df
