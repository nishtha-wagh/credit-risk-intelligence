"""
evaluation/metrics.py

Computes evaluation metrics for RAG output quality.

Metrics:
  - faithfulness      : does the decision cite signals actually present in retrieved context?
  - relevance         : are retrieved chunks relevant to the borrower query?
  - completeness      : does the output cover all major risk dimensions?
  - decision_accuracy : does predicted risk_tier match analyst_risk_tier? (requires labels)
"""

import os
import re
from dataclasses import dataclass

from openai import OpenAI

EVAL_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")   # cheaper model fine for eval scoring


@dataclass
class EvalResult:
    faithfulness: float       # 0.0 – 1.0
    relevance: float          # 0.0 – 1.0
    completeness: float       # 0.0 – 1.0
    decision_accuracy: float | None  # None if no ground truth label
    composite: float          # weighted average

    def to_dict(self) -> dict:
        return self.__dict__


def evaluate(
    decision_output,          # DecisionOutput dataclass
    retrieved_chunks: list,
    ground_truth_tier: str | None = None,
) -> EvalResult:
    client = OpenAI()

    context_text = "\n\n".join(c.text for c in retrieved_chunks)

    faithfulness = _score_faithfulness(client, decision_output, context_text)
    relevance = _score_relevance(client, decision_output.borrower_id, retrieved_chunks)
    completeness = _score_completeness(client, decision_output)

    decision_accuracy = None
    if ground_truth_tier:
        decision_accuracy = 1.0 if decision_output.risk_tier == ground_truth_tier else 0.0

    weights = [0.35, 0.30, 0.25, 0.10] if decision_accuracy is not None else [0.40, 0.35, 0.25]
    scores = [faithfulness, relevance, completeness]
    if decision_accuracy is not None:
        scores.append(decision_accuracy)

    composite = round(sum(w * s for w, s in zip(weights, scores)), 4)

    return EvalResult(
        faithfulness=faithfulness,
        relevance=relevance,
        completeness=completeness,
        decision_accuracy=decision_accuracy,
        composite=composite,
    )


# ---------------------------------------------------------------------------
# LLM-based scorers
# ---------------------------------------------------------------------------

def _score_faithfulness(client, output, context_text: str) -> float:
    prompt = f"""
You are evaluating an AI-generated credit risk assessment for faithfulness.

RETRIEVED CONTEXT:
{context_text[:2000]}

AI OUTPUT:
Decision: {output.decision}
Key signals: {output.key_signals}
Reasoning: {output.reasoning}

Question: Are the key signals and reasoning grounded in the retrieved context above?
Score from 0.0 (completely hallucinated) to 1.0 (fully grounded).
Respond with ONLY a float. No explanation.
""".strip()
    return _ask_score(client, prompt)


def _score_relevance(client, borrower_id: str, chunks: list) -> float:
    chunk_texts = "\n---\n".join(c.text[:200] for c in chunks[:5])
    prompt = f"""
You are evaluating retrieval relevance for a credit risk query about borrower {borrower_id}.

RETRIEVED CHUNKS:
{chunk_texts}

Question: How relevant are these chunks to assessing credit risk?
Score from 0.0 (completely irrelevant) to 1.0 (highly relevant).
Respond with ONLY a float. No explanation.
""".strip()
    return _ask_score(client, prompt)


def _score_completeness(client, output) -> float:
    prompt = f"""
You are evaluating a credit risk assessment for completeness.

OUTPUT:
Risk tier: {output.risk_tier}
Confidence: {output.confidence}
Decision: {output.decision}
Key signals: {output.key_signals}
Reasoning: {output.reasoning}

A complete credit risk assessment should address: payment history, income/DTI, 
credit profile, and a clear action recommendation.

Score from 0.0 (severely incomplete) to 1.0 (fully complete).
Respond with ONLY a float. No explanation.
""".strip()
    return _ask_score(client, prompt)


def _ask_score(client: OpenAI, prompt: str) -> float:
    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"[0-9]+\.?[0-9]*", raw)
    if match:
        return min(1.0, max(0.0, float(match.group())))
    return 0.5
