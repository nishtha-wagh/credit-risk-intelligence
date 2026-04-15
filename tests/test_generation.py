"""
tests/test_generation.py

Unit tests for the generation module.
Mocks LLM calls — no real API calls made.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from generation.generator import _parse_output, generate_assessment
from retrieval.hybrid_retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_LLM_RESPONSE = json.dumps({
    "risk_tier": "HIGH",
    "confidence": 0.82,
    "decision": "Refer to collections review team immediately.",
    "key_signals": [
        "DTI ratio: 0.54 — exceeds policy threshold of 0.45",
        "3 deferred payments in 12-month window",
        "Employment gap: 4 months at origination",
    ],
    "reasoning": (
        "Structured data shows elevated DTI and repeated deferrals. "
        "Retrieved case notes confirm underwriter flagged employment gap. "
        "Pattern consistent with pre-default profile per policy R-114."
    ),
})

SAMPLE_BORROWER = {
    "borrower_id": "B-4821",
    "loan_type": "auto",
    "vintage_year": 2022,
    "fico_score": 618,
    "dti_ratio": 0.54,
    "payments_late_30d": 3,
    "num_deferrals": 3,
    "employment_gap_months": 4,
    "analyst_risk_tier": "HIGH",
    "annual_income": 62000,
}

SAMPLE_CHUNKS = [
    RetrievedChunk(
        chunk_id="N-001_c0",
        borrower_id="B-4821",
        note_id="N-001",
        note_type="underwriter",
        text="Employment gap of 4 months noted. Approved with enhanced monitoring.",
        similarity_score=0.91,
        metadata={},
    )
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_valid_output():
    result = _parse_output(VALID_LLM_RESPONSE)
    assert result["risk_tier"] == "HIGH"
    assert result["confidence"] == 0.82
    assert isinstance(result["key_signals"], list)
    assert len(result["key_signals"]) == 3


def test_parse_output_with_markdown_fences():
    wrapped = f"```json\n{VALID_LLM_RESPONSE}\n```"
    result = _parse_output(wrapped)
    assert result["risk_tier"] == "HIGH"


def test_parse_malformed_output_returns_fallback():
    result = _parse_output("this is not json at all")
    assert result["risk_tier"] == "UNKNOWN"
    assert result["confidence"] == 0.0
    assert "Parse error" in result["decision"]


@patch("generation.generator._call_llm", return_value=VALID_LLM_RESPONSE)
def test_generate_assessment_structure(mock_llm):
    output = generate_assessment("B-4821", SAMPLE_BORROWER, SAMPLE_CHUNKS)
    assert output.borrower_id == "B-4821"
    assert output.risk_tier == "HIGH"
    assert 0.0 <= output.confidence <= 1.0
    assert isinstance(output.key_signals, list)
    assert output.sources_used == 1
    assert output.latency_ms >= 0


@patch("generation.generator._call_llm", return_value=VALID_LLM_RESPONSE)
def test_retrieval_score_computed(mock_llm):
    output = generate_assessment("B-4821", SAMPLE_BORROWER, SAMPLE_CHUNKS)
    assert output.retrieval_score == pytest.approx(0.91, abs=0.01)


@patch("generation.generator._call_llm", return_value=VALID_LLM_RESPONSE)
def test_generate_with_no_chunks(mock_llm):
    output = generate_assessment("B-4821", SAMPLE_BORROWER, [])
    assert output.sources_used == 0
    assert output.retrieval_score == 0.0
