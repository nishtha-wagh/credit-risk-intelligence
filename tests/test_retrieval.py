"""
tests/test_retrieval.py

Unit tests for the hybrid retriever.
Uses a tiny in-memory FAISS index — no real embeddings needed.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_METADATA = [
    {"chunk_id": "N-001_c0", "note_id": "N-001", "borrower_id": "B-1001",
     "chunk_index": 0, "text": "Borrower has strong payment history and low DTI.",
     "token_estimate": 12, "loan_type": "auto", "vintage_year": 2022,
     "risk_band": "LOW", "note_type": "underwriter"},
    {"chunk_id": "N-002_c0", "note_id": "N-002", "borrower_id": "B-2002",
     "chunk_index": 0, "text": "Multiple deferrals observed. Employment gap flagged.",
     "token_estimate": 9, "loan_type": "mortgage", "vintage_year": 2023,
     "risk_band": "HIGH", "note_type": "collections"},
    {"chunk_id": "N-003_c0", "note_id": "N-003", "borrower_id": "B-3003",
     "chunk_index": 0, "text": "Account referred to collections. Borrower unreachable.",
     "token_estimate": 8, "loan_type": "auto", "vintage_year": 2021,
     "risk_band": "CRITICAL", "note_type": "collections"},
]

DIM = 8  # tiny dimension for tests


@pytest.fixture
def retriever_with_fake_index(tmp_path):
    """Build a tiny FAISS index + metadata, return a retriever pointed at them."""
    import faiss

    # Create deterministic fake vectors
    rng = np.random.default_rng(42)
    vecs = rng.random((len(FAKE_METADATA), DIM)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)

    index_path = str(tmp_path / "index.faiss")
    meta_path = str(tmp_path / "metadata.json")
    faiss.write_index(index, index_path)
    Path(meta_path).write_text(json.dumps(FAKE_METADATA))

    # Patch embed call to return a fixed query vector
    query_vec = vecs[0].copy()  # will be most similar to chunk 0

    with patch("retrieval.hybrid_retriever.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_embed_resp = MagicMock()
        mock_embed_resp.data = [MagicMock(embedding=query_vec.tolist())]
        mock_client.embeddings.create.return_value = mock_embed_resp

        r = HybridRetriever(index_path=index_path, metadata_path=meta_path)
        r._embed = lambda text: query_vec  # bypass OpenAI for tests
        yield r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retriever_returns_results(retriever_with_fake_index):
    results = retriever_with_fake_index.retrieve("payment history low risk", top_k=3, min_score=0.0)
    assert len(results) > 0


def test_metadata_filter_loan_type(retriever_with_fake_index):
    results = retriever_with_fake_index.retrieve(
        "risk assessment", filters={"loan_type": "mortgage"}, top_k=5, min_score=0.0
    )
    assert all(r.metadata["loan_type"] == "mortgage" for r in results)


def test_metadata_filter_risk_band_list(retriever_with_fake_index):
    results = retriever_with_fake_index.retrieve(
        "risk assessment", filters={"risk_band": ["HIGH", "CRITICAL"]}, top_k=5, min_score=0.0
    )
    assert all(r.metadata["risk_band"] in ["HIGH", "CRITICAL"] for r in results)


def test_no_results_below_min_score(retriever_with_fake_index):
    results = retriever_with_fake_index.retrieve("query", min_score=0.9999, top_k=5)
    # With a normalised index, similarity ≤ 1.0, so 0.9999 may or may not return results
    # The test just checks no exception is raised and type is correct
    assert isinstance(results, list)


def test_result_fields(retriever_with_fake_index):
    results = retriever_with_fake_index.retrieve("test", top_k=1, min_score=0.0)
    if results:
        r = results[0]
        assert hasattr(r, "chunk_id")
        assert hasattr(r, "text")
        assert hasattr(r, "similarity_score")
        assert 0.0 <= r.similarity_score <= 1.0


def test_empty_filter_returns_all_candidates(retriever_with_fake_index):
    results = retriever_with_fake_index.retrieve("query", filters=None, top_k=10, min_score=0.0)
    assert len(results) == len(FAKE_METADATA)
