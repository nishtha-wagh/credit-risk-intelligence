# Decision-Aware RAG System — Credit Risk Intelligence

> A production-style Retrieval-Augmented Generation system that combines structured credit features with unstructured borrower narratives to generate **decision-ready risk assessments** — not generic summaries.

---

## Problem

Credit analysts spend hours manually reconciling structured loan data (DTI ratios, FICO scores, payment history) with unstructured case notes, call transcripts, and underwriter comments. Traditional RAG systems return text summaries. This system returns **structured, auditable decisions** with confidence scores — designed to slot directly into a risk workflow.

## Demo

![Demo](docs/assets/credit-risk-intelligence-demo.gif)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│   Borrower ID  →  Structured Signals + Unstructured Text        │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       HYBRID RETRIEVAL      │
          │  ┌─────────┐  ┌──────────┐ │
          │  │Metadata │  │  Vector  │ │
          │  │ Filter  │  │ Search   │ │
          │  │         │  │ (FAISS)  │ │
          │  └────┬────┘  └────┬─────┘ │
          │       └─────┬──────┘       │
          │    Deduped & Re-ranked      │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      ML SCORING LAYER       │
          │  XGBoost Classifier         │
          │  + SHAP Feature Attribution │
          │  → risk probability signal  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     STRUCTURED GENERATION   │
          │  Prompt Template            │
          │  + Context Window Assembly  │
          │  → LLM (Groq / OpenAI /     │
          │         Ollama / Anthropic) │
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

### 1. Hybrid Retrieval (not plain vector search)
Standard RAG retrieves by semantic similarity alone. This system first applies a **metadata pre-filter** (e.g. loan type, risk band, vintage year) to constrain the search space, then runs vector similarity over the filtered subset. Results are deduplicated by both `note_id` and text fingerprint to prevent the same note filling all top-k slots. This eliminates irrelevant and redundant context, improving precision for structured domains.

### 2. Structured Output (not free-text summaries)
The LLM is instructed to return a strict JSON schema — `risk_tier`, `confidence`, `decision`, `key_signals`, `reasoning`. This makes outputs **auditable, parseable, and actionable** in downstream systems.

### 3. ML Scoring Layer (XGBoost + SHAP)
An XGBoost binary classifier runs alongside the RAG pipeline, trained on the same borrower features (FICO, DTI, payment history, deferrals, etc.). Its predicted tier and default probability are injected into the LLM context as an additional signal, and SHAP values provide per-feature attribution for explainability. The ML layer is optional and togglable from the UI.

### 4. Multi-Provider LLM Support
The system routes to any of four LLM backends via a single `LLM_PROVIDER` env variable — Groq (free, fast, recommended), Ollama (fully local, no API key), OpenAI, or Anthropic. Embeddings similarly support Ollama (free, local) or OpenAI. This makes the system runnable at zero cost.

### 5. Separation of Concerns
Each module (ingestion, retrieval, generation, evaluation) is independent and testable in isolation — designed for easy extension (swap FAISS for BigQuery Vector Search, swap Groq for Claude, etc.).

---

## Repo Structure

```
rag-credit-risk/
├── data/
│   ├── raw/                    # Source CSVs (borrowers.csv, case_notes.csv)
│   ├── processed/              # FAISS index, metadata JSON, XGBoost model
│   └── schemas/                # BigQuery-compatible SQL schemas
│
├── ingestion/
│   ├── loader.py               # Load + validate raw borrower data
│   ├── chunker.py              # Text chunking strategies
│   └── embedder.py             # Embed text → vectors, store in FAISS
│                               # Supports: Ollama (free) or OpenAI
│
├── retrieval/
│   └── hybrid_retriever.py     # Metadata pre-filter + FAISS vector search
│                               # Deduplication by note_id + text fingerprint
│
├── generation/
│   ├── context_builder.py      # Assemble retrieved docs into context window
│   ├── generator.py            # LLM routing + structured output parsing
│                               # Supports: Groq, Ollama, OpenAI, Anthropic
│   ├── xgb_scorer.py           # XGBoost classifier (train + score)
│   └── shap_explainer.py       # SHAP feature attribution
│
├── prompts/
│   ├── system_prompt.txt       # System-level LLM instructions
│   └── decision_template.txt   # Decision generation prompt template
│
├── evaluation/
│   ├── metrics.py              # Faithfulness, relevance, completeness (LLM-judged)
│   └── run_eval.py             # Batch evaluation runner with MLflow support
│
├── api/
│   └── main.py                 # FastAPI app (/assess, /batch, /health)
│
├── app/
│   ├── streamlit_app.py        # 3-panel decision system UI (system-themed)
│   ├── utils/shared.py         # Cached loaders, constants, CSS helpers
│   └── pages/
│       ├── 1_🎯_Live_Assessor.py       # Per-borrower AI assessment
│       ├── 2_🔎_Retrieval_Explorer.py  # FAISS retrieval inspection
│       ├── 3_📈_Eval_Dashboard.py      # Evaluation metrics dashboard
│       └── 4_🧩_Model_Explainability.py # SHAP waterfall charts
│
├── scripts/
│   ├── generate_mock_data.py   # Generate synthetic borrower dataset (200 records)
│   ├── build_index.py          # Chunk + embed + build FAISS index
│   └── train_xgb.py            # Train + save XGBoost model
│
├── tests/
│   ├── test_retrieval.py
│   └── test_generation.py
│
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/rag-credit-risk.git
cd rag-credit-risk
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your provider of choice:

**Option A — Groq (recommended: free, fast, no GPU)**
```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=gsk_your_groq_key_here     # get from console.groq.com
OPENAI_BASE_URL=https://api.groq.com/openai/v1

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

**Option B — Fully local (no API key, needs Ollama)**
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
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start Ollama (if using Ollama embeddings)

```bash
ollama serve                      # keep running in a background terminal
ollama pull nomic-embed-text      # one-time download (~274MB)
```

### 4. Generate data, build index, train model

```bash
# Export env vars first (macOS/Linux)
export $(grep -v '^#' .env | grep -v '^$' | xargs)

python scripts/generate_mock_data.py   # → data/raw/borrowers.csv + case_notes.csv
python scripts/build_index.py          # → data/processed/index.faiss + metadata.json
python scripts/train_xgb.py            # → data/processed/xgb_model.json
```

### 5. Run the Streamlit app

```bash
export $(grep -v '^#' .env | grep -v '^$' | xargs) && streamlit run app/streamlit_app.py
```

### 6. Run batch evaluation

```bash
export $(grep -v '^#' .env | grep -v '^$' | xargs) && python -m evaluation.run_eval --n 20
# With XGBoost signal:
python -m evaluation.run_eval --xgb --n 20
```

### 7. Run the API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### 8. Docker (full stack)

```bash
docker-compose up --build
```

---

## Example Output

**Input:** Borrower ID `B-4821`, loan type `auto`, vintage `2022`

**Output:**
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
  "reasoning": "Structured data indicates elevated DTI and payment irregularity. Retrieved case notes (similarity: 0.66) confirm underwriter flagged employment instability at origination. Risk tier set to HIGH based on policy rule R-114.",
  "retrieval_score": 0.66,
  "sources_used": 5,
  "xgb_predicted_tier": "HIGH",
  "xgb_probability": 0.81
}
```

---

## Evaluation Results

Evaluated on 20-record holdout set using LLM-as-judge scoring (Groq llama-3.3-70b-versatile).

| Metric | Score |
|---|---|
| Faithfulness | 0.50 |
| Relevance | 0.80 |
| Completeness | 0.92 |
| Decision accuracy (vs. analyst labels) | 0.40 |
| Composite score | 0.685 |
| Avg confidence | 0.91 |
| Avg latency (single assessment) | ~1.5s |

> Note: Decision accuracy of 0.40 reflects LLM vs. analyst label agreement on a synthetic dataset with 86% HIGH/CRITICAL class imbalance — not a calibrated benchmark. Faithfulness and completeness are the more meaningful metrics for this system.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/assess` | POST | Single borrower risk assessment |
| `/batch` | POST | Batch assessment (up to 50 borrowers) |
| `/health` | GET | Service health check |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Ollama `nomic-embed-text` (free, local) or OpenAI `text-embedding-3-small` |
| Vector store | FAISS (in-memory, local) |
| LLM | Groq `llama-3.3-70b-versatile` (default) · Ollama · OpenAI · Anthropic |
| ML scoring | XGBoost classifier + SHAP explainer |
| API | FastAPI |
| UI | Streamlit (3-panel decision system, system-themed dark/light) |
| Evaluation | LLM-as-judge (faithfulness, relevance, completeness) |
| Data | BigQuery-compatible CSV schema |
| Containerisation | Docker + Docker Compose |
| Testing | pytest |

---

## Limitations

- Mock dataset is synthetic — production deployment requires real borrower data governance
- `export $(grep -v '^#' .env | grep -v '^$' | xargs)` must be re-run each terminal session (or add `load_dotenv()` to scripts)
- FAISS index held in memory — swap to BigQuery Vector Search or Pinecone for scale
- XGBoost trained on mock data; accuracy metrics are illustrative, not production-calibrated
- LLM outputs require human-in-the-loop review before operational use

---

## Extensions

- [x] XGBoost score as structured signal alongside RAG output
- [x] SHAP values on structured features for explainability layer
- [x] MLflow experiment tracking for retrieval + generation metrics
- [ ] Swap FAISS → BigQuery Vector Search for GCP-native deployment
- [ ] Fine-tune embedding model on credit domain vocabulary
- [ ] Calibrate confidence scoring against actuarial benchmarks
- [ ] Add streaming LLM responses for lower perceived latency
