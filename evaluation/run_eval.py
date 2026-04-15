"""
evaluation/run_eval.py

Batch evaluation runner with MLflow experiment tracking.

Runs assessments on a holdout set, scores each with evaluation metrics,
logs everything to MLflow, and prints a summary report.

Run (no MLflow UI):
    python -m evaluation.run_eval --n 20

Run (with MLflow UI):
    mlflow ui                              # → http://localhost:5000
    python -m evaluation.run_eval --n 20 --experiment "rag-credit-risk-v1"
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import evaluate
from generation.generator import generate_assessment
from retrieval.hybrid_retriever import HybridRetriever

# Optional MLflow — gracefully degrade if not installed
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def main(n, output_path, experiment_name, run_name, use_xgb, top_k):
    print("=== EvalBench: Credit Risk RAG ===\n")

    borrowers_df = pd.read_csv("data/raw/borrowers.csv")
    holdout = borrowers_df.sample(n=min(n, len(borrowers_df)), random_state=99)
    retriever = HybridRetriever()

    scorer = None
    explainer = None
    if use_xgb:
        from generation.xgb_scorer import CreditScorer
        from generation.shap_explainer import SHAPExplainer
        scorer = CreditScorer()
        try:
            scorer.load()
            print("[eval] XGBoost model loaded from disk")
        except Exception:
            print("[eval] Training XGBoost on full dataset...")
            scorer.train(borrowers_df, save=True)
        try:
            explainer = SHAPExplainer(scorer)
            print("[eval] SHAP explainer ready")
        except Exception as e:
            print(f"[eval] SHAP unavailable: {e}")

    run_params = {"n_samples": n, "top_k": top_k,
                  "xgb_enabled": use_xgb, "shap_enabled": explainer is not None}

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(run_params)
        print(f"[mlflow] Run started: {active_run.info.run_id}")
    else:
        active_run = None
        print("[eval] MLflow not installed — tracking skipped (pip install mlflow)")

    results = []
    t_start = time.time()

    for _, row in tqdm(holdout.iterrows(), total=len(holdout), desc="Evaluating"):
        bid = row["borrower_id"]
        ground_truth = row.get("analyst_risk_tier")
        borrower_row = row.to_dict()

        filters = {"loan_type": borrower_row.get("loan_type")}
        chunks = retriever.retrieve(
            query=f"credit risk signals for borrower {bid}",
            filters=filters, top_k=top_k,
        )

        xgb_signal = scorer.score(borrower_row) if scorer else None
        shap_signal = explainer.explain(borrower_row) if explainer else None

        output = generate_assessment(
            bid, borrower_row, chunks,
            xgb_signal=xgb_signal, shap_signal=shap_signal,
        )
        eval_result = evaluate(output, chunks, ground_truth_tier=ground_truth)

        results.append({
            "borrower_id":        bid,
            "predicted_tier":     output.risk_tier,
            "ground_truth_tier":  ground_truth,
            "xgb_predicted_tier": output.xgb_predicted_tier,
            "xgb_probability":    output.xgb_probability,
            "confidence":         output.confidence,
            "retrieval_score":    output.retrieval_score,
            "sources_used":       output.sources_used,
            "latency_ms":         output.latency_ms,
            **eval_result.to_dict(),
        })

    total_time = time.time() - t_start
    df = pd.DataFrame(results)

    metrics = {
        "faithfulness":        df["faithfulness"].mean(),
        "relevance":           df["relevance"].mean(),
        "completeness":        df["completeness"].mean(),
        "decision_accuracy":   df["decision_accuracy"].mean(),
        "composite_score":     df["composite"].mean(),
        "avg_confidence":      df["confidence"].mean(),
        "avg_retrieval_score": df["retrieval_score"].mean(),
        "avg_latency_ms":      df["latency_ms"].mean(),
        "total_eval_time_s":   round(total_time, 1),
        "n_evaluated":         len(df),
    }
    if use_xgb and df["xgb_predicted_tier"].notna().any():
        metrics["xgb_accuracy"] = (df["xgb_predicted_tier"] == df["ground_truth_tier"]).mean()

    if MLFLOW_AVAILABLE and active_run:
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)
        if scorer and scorer._model:
            mlflow.xgboost.log_model(scorer._model, artifact_path="xgb_model")
        mlflow.end_run()
        print("\n[mlflow] Run logged. View at: mlflow ui")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    _print_summary(metrics)
    print(f"\n✅ Results saved → {output_path}")


def _print_summary(metrics):
    print("\n" + "=" * 44)
    print("  EVALUATION SUMMARY")
    print("=" * 44)
    rows = [
        ("Samples evaluated",    f"{int(metrics['n_evaluated'])}"),
        ("Faithfulness",         f"{metrics['faithfulness']:.3f}"),
        ("Relevance",            f"{metrics['relevance']:.3f}"),
        ("Completeness",         f"{metrics['completeness']:.3f}"),
        ("Decision accuracy",    f"{metrics['decision_accuracy']:.3f}"),
        ("Composite score",      f"{metrics['composite_score']:.3f}"),
        ("Avg confidence",       f"{metrics['avg_confidence']:.3f}"),
        ("Avg retrieval score",  f"{metrics['avg_retrieval_score']:.3f}"),
        ("Avg latency",          f"{metrics['avg_latency_ms']:.0f}ms"),
        ("Total time",           f"{metrics['total_eval_time_s']}s"),
    ]
    if "xgb_accuracy" in metrics:
        rows.append(("XGBoost accuracy", f"{metrics['xgb_accuracy']:.3f}"))
    for label, value in rows:
        print(f"  {label:<26s}: {value}")
    print("=" * 44)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=20)
    parser.add_argument("--output",     type=str, default="docs/evaluation_results.csv")
    parser.add_argument("--experiment", type=str, default="rag-credit-risk")
    parser.add_argument("--run-name",   type=str, default="eval-run")
    parser.add_argument("--xgb",        action="store_true")
    parser.add_argument("--top-k",      type=int, default=5)
    args = parser.parse_args()
    main(args.n, args.output, args.experiment, args.run_name, args.xgb, args.top_k)
