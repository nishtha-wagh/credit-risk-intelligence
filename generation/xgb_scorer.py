"""
generation/xgb_scorer.py

Trains a lightweight XGBoost classifier on structured borrower features
and returns a calibrated risk probability + predicted tier.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from generation.context_builder import XGBoostSignal

FEATURES = [
    "fico_score", "dti_ratio", "ltv_ratio",
    "payments_late_30d", "payments_late_60d", "payments_late_90d",
    "num_deferrals", "employment_gap_months", "num_open_accounts",
    "credit_history_yrs", "annual_income", "loan_amount", "loan_term_months",
]

TIER_ORDER      = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
HIGH_RISK_TIERS = {"HIGH", "CRITICAL"}
MODEL_PATH      = Path(os.getenv("XGB_MODEL_PATH", "data/processed/xgb_model.json"))


class CreditScorer:
    def __init__(self):
        self._model    = None   # XGBClassifier (after train)
        self._booster  = None   # xgb.Booster   (after load)
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, save: bool = True) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Run: pip install xgboost")

        X, y = self._prepare(df)

        self._model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
        self._model.fit(X, y)
        self._booster    = self._model.get_booster()
        self._is_trained = True
        print(f"[xgb_scorer] Trained on {len(df)} samples")

        if save:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._booster.save_model(str(MODEL_PATH))
            print(f"[xgb_scorer] Model saved → {MODEL_PATH}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path = MODEL_PATH) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Run: pip install xgboost")
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))
        self._model      = None   # sklearn wrapper not available in load-only mode
        self._is_trained = True
        print(f"[xgb_scorer] Model loaded from {path}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, borrower_row: dict) -> XGBoostSignal:
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        import xgboost as xgb
        row_df = self._row_to_df(borrower_row)

        if self._booster is not None:
            # Works whether loaded from disk or trained
            dmatrix       = xgb.DMatrix(row_df)
            prob_high_risk = float(self._booster.predict(dmatrix)[0])
        else:
            prob_high_risk = float(self._model.predict_proba(row_df)[0][1])

        predicted_tier = _prob_to_tier(prob_high_risk)
        class_probs    = _binary_prob_to_class_probs(prob_high_risk)

        return XGBoostSignal(
            predicted_tier=predicted_tier,
            probability=round(prob_high_risk, 4),
            class_probabilities={k: round(v, 4) for k, v in class_probs.items()},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df[FEATURES].copy().fillna(df[FEATURES].median(numeric_only=True))
        y = df["analyst_risk_tier"].map(lambda t: 1 if t in HIGH_RISK_TIERS else 0)
        return X, y

    def _row_to_df(self, row: dict) -> pd.DataFrame:
        data = {f: [row.get(f, np.nan)] for f in FEATURES}
        df   = pd.DataFrame(data)
        return df.fillna(df.median(numeric_only=True))

    @property
    def feature_names(self) -> list[str]:
        return FEATURES


# ---------------------------------------------------------------------------
# Tier mapping helpers
# ---------------------------------------------------------------------------

def _prob_to_tier(p: float) -> str:
    if p >= 0.75: return "CRITICAL"
    if p >= 0.50: return "HIGH"
    if p >= 0.25: return "MEDIUM"
    return "LOW"


def _binary_prob_to_class_probs(p_high: float) -> dict[str, float]:
    p_low = 1.0 - p_high
    return {
        "LOW":      p_low  * 0.6,
        "MEDIUM":   p_low  * 0.4,
        "HIGH":     p_high * 0.55,
        "CRITICAL": p_high * 0.45,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/raw/borrowers.csv")
    scorer = CreditScorer()
    scorer.train(df, save=True)
    print("Done.")
