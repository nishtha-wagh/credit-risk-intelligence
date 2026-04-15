# Architecture: Decision-Aware RAG System

## Design Principles

### 1. Decision-ready output, not summaries
Standard RAG returns prose. This system enforces a strict JSON output schema with `risk_tier`, `confidence`, `decision`, `key_signals`, and `reasoning`. Every field is auditable and parseable by downstream systems.

### 2. Hybrid retrieval
Pre-filter by structured metadata (loan type, vintage year, risk band) before running vector similarity search. This reduces noise and improves precision — particularly important in credit risk where irrelevant context can distort LLM outputs.

### 3. Separation of concerns
Each layer (ingestion → retrieval → generation → evaluation) is independently testable and swappable. Changing from FAISS to BigQuery Vector Search, or from OpenAI to Claude, requires changing one file.

### 4. Confidence scoring
Confidence is derived from retrieval score (average cosine similarity of retrieved chunks) and LLM consistency signals. Low confidence flags cases for human review.

---

## Data Flow

```
generate_mock_data.py
        ↓
data/raw/borrowers.csv + case_notes.csv
        ↓
ingestion/loader.py  →  ingestion/chunker.py  →  ingestion/embedder.py
        ↓
data/processed/index.faiss + metadata.json
        ↓
retrieval/hybrid_retriever.py
  └── Step 1: metadata_filter (loan_type, vintage_year, risk_band)
  └── Step 2: vector_search (FAISS IndexFlatIP, cosine similarity)
  └── Step 3: re-rank + threshold (min_score filter)
        ↓
generation/generator.py
  └── context_builder (structured signals + retrieved chunks)
  └── LLM call (OpenAI / Anthropic)
  └── JSON parser + confidence scorer
        ↓
DecisionOutput (risk_tier, confidence, decision, key_signals, reasoning)
        ↓
  ┌─────────────┐
  │  FastAPI     │  ← /assess, /batch endpoints
  └─────────────┘
  ┌─────────────┐
  │  Streamlit   │  ← interactive demo UI
  └─────────────┘
```

---

## Chunking Strategy

- Chunk size: ~300 tokens (1,200 characters)
- Overlap: 50 tokens (200 characters)
- Rationale: credit case notes are short (100–400 words). Overlap preserves context at chunk boundaries. No semantic splitting needed at this scale.

## Embedding Model

- `text-embedding-3-small` (OpenAI): 1536 dimensions, strong performance on domain text, low cost.
- Vectors L2-normalised before storage → inner product = cosine similarity.

## Retrieval Parameters

| Parameter | Default | Notes |
|---|---|---|
| `top_k` | 5 | Chunks passed to LLM context |
| `min_score` | 0.70 | Below this → chunk excluded |
| Filter field | `loan_type` | Auto-injected from borrower record |

## LLM Configuration

- `temperature: 0.0` for deterministic outputs
- `response_format: json_object` (OpenAI) to enforce schema
- System prompt explicitly forbids hallucination and requires evidence citation

---

## Extension Points

| Extension | Where to change |
|---|---|
| Swap FAISS → Pinecone | `retrieval/hybrid_retriever.py` |
| Swap FAISS → BQ Vector Search | `retrieval/hybrid_retriever.py` |
| Add XGBoost score as signal | `generation/context_builder.py` |
| Add SHAP values | `generation/context_builder.py` |
| Add MLflow tracking | `evaluation/run_eval.py` |
| Fine-tune embeddings | `ingestion/embedder.py` |
