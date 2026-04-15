"""
api/main.py  +  api/routes.py  (combined for simplicity)

Endpoints:
  POST /assess        — single borrower risk assessment
  POST /batch         — batch assessments (up to 50)
  GET  /health        — service health check
"""

import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation.generator import generate_assessment
from retrieval.hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AssessRequest(BaseModel):
    borrower_id: str
    filters: dict | None = Field(default=None, description="Metadata filters for hybrid retrieval")
    top_k: int = Field(default=5, ge=1, le=20)


class AssessResponse(BaseModel):
    borrower_id: str
    risk_tier: str
    confidence: float
    decision: str
    key_signals: list[str]
    reasoning: str
    retrieval_score: float
    sources_used: int
    latency_ms: int


class BatchRequest(BaseModel):
    borrower_ids: list[str] = Field(..., max_length=50)
    filters: dict | None = None
    top_k: int = 5


class HealthResponse(BaseModel):
    status: str
    index_vectors: int
    model: str


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

retriever: HybridRetriever | None = None
borrowers_df: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, borrowers_df
    retriever = HybridRetriever()
    borrowers_df = pd.read_csv("data/raw/borrowers.csv")
    yield


app = FastAPI(
    title="Credit Risk RAG API",
    description="Decision-aware RAG system for credit risk assessment",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        index_vectors=retriever.index.ntotal if retriever else 0,
        model=os.getenv("LLM_MODEL", "gpt-4o"),
    )


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest):
    borrower_row = _get_borrower(req.borrower_id)

    # Auto-inject loan_type filter if not overridden
    filters = req.filters or {"loan_type": borrower_row.get("loan_type")}

    chunks = retriever.retrieve(
        query=f"credit risk assessment for borrower {req.borrower_id}",
        filters=filters,
        top_k=req.top_k,
    )

    result = generate_assessment(req.borrower_id, borrower_row, chunks)

    return AssessResponse(
        borrower_id=result.borrower_id,
        risk_tier=result.risk_tier,
        confidence=result.confidence,
        decision=result.decision,
        key_signals=result.key_signals,
        reasoning=result.reasoning,
        retrieval_score=result.retrieval_score,
        sources_used=result.sources_used,
        latency_ms=result.latency_ms,
    )


@app.post("/batch", response_model=list[AssessResponse])
def batch_assess(req: BatchRequest):
    return [
        assess(AssessRequest(
            borrower_id=bid,
            filters=req.filters,
            top_k=req.top_k,
        ))
        for bid in req.borrower_ids
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_borrower(borrower_id: str) -> dict:
    row = borrowers_df[borrowers_df["borrower_id"] == borrower_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Borrower {borrower_id} not found")
    return row.iloc[0].to_dict()
