# Credit Risk Intelligence — Decision-Aware RAG System

> A production-style Retrieval-Augmented Generation system that combines structured credit features with unstructured borrower narratives to generate **decision-ready risk assessments**, not generic summaries.

<p align="center">
  <img src="docs/assets/credit-risk-intelligence-demo.gif" alt="Demo" width="900"/>
</p>

---

## Problem

Credit analysts spend hours manually reconciling structured loan data (DTI ratios, FICO scores, payment history) with unstructured case notes, call transcripts, and underwriter comments. Traditional RAG systems return text summaries. This system returns **structured, auditable decisions** with confidence scores — designed to slot directly into a risk workflow.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│   Borrower ID  →  Structured Signals + Unstructured Text        │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       HYBRID RETRIEVAL      │
          │  Metadata pre-filter        │
          │  + FAISS vector search      │
          │  + Dedup & re-rank          │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      ML SCORING LAYER       │
          │  XGBoost (13 features)      │
          │  + SHAP attribution         │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     STRUCTURED GENERATION   │
          │  Context assembly           │
          │  → Groq / OpenAI /          │
          │    Ollama / Anthropic       │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      DECISION OUTPUT        │
          │  {                          │
          │    "risk_tier": "HIGH",     │
          │    "confidence": 0.87,      │
          │    "decision": "...",       │
          │    "key_signals": [...],    │
          │    "reasoning": "..."       │
          │  }                          │
          └─────────────────────────────┘
```

---

## Key Design Decisions

**Hybrid retrieval over plain vector search** — metadata pre-filter narrows the candidate pool by loan type and risk band before running FAISS similarity search. Results are deduplicated by `note_id` and text fingerprint to prevent the same note filling all top-k slots.

**Structured JSON output over free-text summaries** — the LLM is constrained to a strict schema (`risk_tier`, `confidence`, `decision`, `key_signals`, `reasoning`), making outputs auditable and parseable by downstream systems.

**XGBoost + SHAP as a grounding signal** — a binary classifier trained on 13 structured credit features runs in parallel with retrieval. Its risk probability and SHAP feature attributions are injected into the LLM context, giving the model a calibrated ML signal to reason against.

**Multi-provider LLM routing** — `LLM_PROVIDER` in `.env` switches between Groq (free, fast), Ollama (fully local), OpenAI, or Anthropic with no code changes. Same pattern for embeddings.

---

## Repo Structure

```
rag-credit-risk/
├── data/
│   ├── raw/                    # borrowers.csv, case_notes.csv (200 borrowers, 718 notes)
│   ├── processed/              # index.faiss, metadata.json, xgb_model.json
│   └── schemas/                # BigQuery-compatible SQL schemas
│
├── ingestion/
│   ├── loader.py               # Load + validate raw borrower data
│   ├── chunker.py              # 300-token chunks, 50-token overlap
│   └── embedder.py             # Ollama or OpenAI embeddings → FAISS
│
├── retrieval/
│   └── hybrid_retriever.py     # Metadata filter + vector search + deduplication
│
├── generation/
│   ├── context_builder.py      # Structured signals + chunks + XGBoost signal
│   ├── generator.py            # LLM routing (Groq/Ollama/OpenAI/Anthropic)
│   ├── xgb_scorer.py           # XGBoost train + score (booster-based persistence)
│   └── shap_explainer.py       # SHAP TreeExplainer, top-N feature attribution
│
├── prompts/
│   ├── system_prompt.txt
│   └── decision_template.txt
│
├── evaluation/
│   ├── metrics.py              # LLM-as-judge: faithfulness, relevance, completeness
│   └── run_eval.py             # Batch eval runner, optional MLflow tracking
│
├── api/
│   └── main.py                 # FastAPI: /assess, /batch, /health
│
├── app/
│   ├── streamlit_app.py        # 3-panel decision UI (system dark/light theme)
│   ├── utils/shared.py         # Cached loaders, CSS, constants
│   └── pages/
│       ├── 1_🎯_Live_Assessor.py
│       ├── 2_🔎_Retrieval_Explorer.py
│       ├── 3_📈_Eval_Dashboard.py
│       └── 4_🧩_Model_Explainability.py
│
├── scripts/
│   ├── generate_mock_data.py   # Synthetic dataset (200 borrowers, 718 case notes)
│   ├── build_index.py          # Chunk + embed + store FAISS index
│   └── train_xgb.py            # Train + save XGBoost model
│
├── tests/
│   ├── test_retrieval.py
│   └── test_generation.py
│
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/nishtha-wagh/credit-risk-intelligence.git
cd credit-risk-intelligence
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

**Option A — Groq (recommended: free, fast)**
```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=gsk_your_key    # console.groq.com → API Keys
OPENAI_BASE_URL=https://api.groq.com/openai/v1
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

**Option B — Fully local (no API key)**
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

**Option C — OpenAI**
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start Ollama (if using Ollama embeddings)

```bash
ollama serve                       # keep running in background
ollama pull nomic-embed-text       # one-time, ~274MB
```

### 4. Generate data, build index, train model

```bash
export $(grep -v '^#' .env | grep -v '^$' | xargs)

python scripts/generate_mock_data.py   # → data/raw/
python scripts/build_index.py          # → data/processed/index.faiss
python scripts/train_xgb.py            # → data/processed/xgb_model.json
```

### 5. Launch

```bash
# Streamlit UI
export $(grep -v '^#' .env | grep -v '^$' | xargs) && streamlit run app/streamlit_app.py

# Batch evaluation
python -m evaluation.run_eval --xgb --n 20

# FastAPI
uvicorn api.main:app --reload   # → http://localhost:8000/docs

# Docker
docker-compose up --build
```

---

## Example Output

```json
{
  "borrower_id": "B-4821",
  "risk_tier": "HIGH",
  "confidence": 0.84,
  "decision": "Recommend referral to collections review. Pattern of deferred payments combined with recent employment gap exceeds policy threshold.",
  "key_signals": [
    "3 deferred payments in 12-month window",
    "Employment gap: 4 months (flagged in underwriter note)",
    "DTI ratio: 0.54 (above 0.45 threshold)",
    "No prior default — mitigating factor"
  ],
  "reasoning": "Structured data indicates elevated DTI and payment irregularity. Retrieved case notes (similarity: 0.66) confirm underwriter flagged employment instability at origination.",
  "retrieval_score": 0.66,
  "sources_used": 5,
  "xgb_predicted_tier": "HIGH",
  "xgb_probability": 0.81
}
```

---

## Evaluation Results

Evaluated on 20-record holdout · LLM-as-judge via Groq llama-3.3-70b-versatile

| Metric | Score |
|---|---|
| Faithfulness | 0.50 |
| Relevance | 0.80 |
| Completeness | 0.92 |
| Decision accuracy (vs. analyst labels) | 0.40 |
| Composite score | 0.685 |
| Avg confidence | 0.91 |
| Avg latency | ~1.5s |

> Decision accuracy of 0.40 reflects class imbalance in the synthetic dataset (86% HIGH/CRITICAL) — not a calibrated production benchmark. Completeness (0.92) and relevance (0.80) are the more meaningful signal quality metrics.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Ollama `nomic-embed-text` (free) or OpenAI `text-embedding-3-small` |
| Vector store | FAISS `IndexFlatIP` (cosine similarity, in-memory) |
| LLM | Groq `llama-3.3-70b-versatile` · Ollama · OpenAI · Anthropic |
| ML scoring | XGBoost binary classifier + SHAP TreeExplainer |
| API | FastAPI |
| UI | Streamlit — 3-panel, system dark/light theme |
| Evaluation | LLM-as-judge (faithfulness · relevance · completeness) |
| Containerisation | Docker + Docker Compose |
| Testing | pytest |

---

## Limitations

- Synthetic dataset — production use requires real data governance and model recalibration
- `.env` vars must be re-exported each terminal session (or add `load_dotenv()` to scripts)
- FAISS is in-memory — swap to Pinecone or BigQuery Vector Search beyond ~1M vectors
- LLM outputs require human-in-the-loop review before operational use

---

## Extensions

- [x] XGBoost score injected as structured signal into LLM context
- [x] SHAP feature attribution for per-decision explainability
- [x] MLflow experiment tracking (optional, `pip install mlflow`)
- [ ] BigQuery Vector Search for GCP-native deployment
- [ ] Fine-tune embedding model on credit domain vocabulary
- [ ] Streaming LLM responses for lower perceived latency
- [ ] Confidence calibration against actuarial benchmarks
