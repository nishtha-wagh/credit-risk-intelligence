"""
scripts/train_xgb.py

Train the XGBoost credit risk scorer and persist it to disk.
Run this once after generating mock data and building the FAISS index.

Run:
    python scripts/train_xgb.py [--eval]

Flags:
    --eval   Run held-out evaluation and print classification report
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.xgb_scorer import CreditScorer, FEATURES


def main(run_eval: bool):
    print("=== Training XGBoost Credit Risk Scorer ===\n")

    df = pd.read_csv("data/raw/borrowers.csv")
    print(f"Dataset: {len(df)} borrowers")

    tier_dist = df["analyst_risk_tier"].value_counts()
    print("\nRisk tier distribution:")
    for tier, count in tier_dist.items():
        print(f"  {tier:<10s}: {count} ({count/len(df)*100:.0f}%)")

    if run_eval:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, roc_auc_score
        import xgboost as xgb
        import numpy as np

        HIGH_RISK = {"HIGH", "CRITICAL"}
        X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
        y = df["analyst_risk_tier"].map(lambda t: 1 if t in HIGH_RISK else 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("\n--- Held-out Evaluation ---")
        print(classification_report(y_test, y_pred, target_names=["LOW/MEDIUM", "HIGH/CRITICAL"]))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

        # Feature importance
        importance = sorted(
            zip(FEATURES, model.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True
        )
        print("\nTop feature importances:")
        for feat, imp in importance[:8]:
            bar = "█" * int(imp * 50)
            print(f"  {feat:<30s} {imp:.4f}  {bar}")

    # Train on full data and save
    scorer = CreditScorer()
    scorer.train(df, save=True)
    print("\n✅ Model ready. Run evaluations with:")
    print("   python -m evaluation.run_eval --xgb --n 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run held-out evaluation before saving")
    args = parser.parse_args()
    main(args.eval)
