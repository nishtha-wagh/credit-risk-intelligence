# Architecture — Credit Risk Decision System

> Implementation details for engineers. For an overview of what the system does and how to run it, see the README.

---

## Data Flow

```
generate_mock_data.py
        ↓
data/raw/borrowers.csv          (200 borrowers, 13 structured features + label)
data/raw/case_notes.csv         (718 case notes: underwriter, collections, servicing, complaint)
        ↓
ingestion/loader.py  →  chunker.py  →  embedder.py
        ↓
data/processed/index.faiss      (718 vectors, 768-dim, L2-normalised)
data/processed/metadata.json    (parallel array of chunk metadata)
data/processed/xgb_model.json   (XGBoost booster, saved via get_booster().save_model())
        ↓
retrieval/hybrid_retriever.py
  ├── Step 1: metadata pre-filter (loan_type, risk_band, vintage_year)
  ├── Step 2: FAISS cosine similarity over filtered subset
  └── Step 3: dedup by note_id + text fingerprint, re-rank, threshold
        ↓                              ↓
generation/xgb_scorer.py         generation/generator.py
  └── predict_proba → tier             └── context_builder.py
  └── SHAP top-N features                  (signals + chunks + XGBoost signal)
        ↓                              ↓
                    LLM call (temp=0.0)
                    Groq / Ollama / OpenAI / Anthropic
                        ↓
                   JSON parse + fallback
                        ↓
                   DecisionOutput dataclass
                   ┌──────────┐   ┌───────────┐
                   │ FastAPI  │   │ Streamlit │
                   └──────────┘   └───────────┘
```

---

## Layer 1 — Ingestion

### Data sources
| File | Rows | Key fields |
|---|---|---|
| `borrowers.csv` | 200 | fico_score, dti_ratio, ltv_ratio, payments_late_30/60/90d, num_deferrals, employment_gap_months, annual_income, loan_amount, loan_term_months, credit_history_yrs, num_open_accounts, loan_type, vintage_year, analyst_risk_tier |
| `case_notes.csv` | 718 | borrower_id, loan_id, note_type, note_date, author_role, note_text, loan_type, risk_band |

### Chunking (`ingestion/chunker.py`)
- Chunk size: 300 tokens (~1,200 characters)
- Overlap: 50 tokens — preserves context at chunk boundaries
- Each chunk inherits parent note metadata (`borrower_id`, `note_id`, `note_type`, `loan_type`)
- No semantic splitting needed at this dataset scale; fixed-size with overlap is sufficient

### Embedding (`ingestion/embedder.py`)
- Vectors are L2-normalised before storage → inner product equals cosine similarity
- FAISS `IndexFlatIP` (exact search, no approximation) — appropriate for 718 vectors
- Two providers, set via `EMBEDDING_PROVIDER`:

| Provider | Model | Dimensions | Cost |
|---|---|---|---|
| `ollama` (default) | `nomic-embed-text` | 768 | Free (local) |
| `openai` | `text-embedding-3-small` | 1536 | ~$0.01/run |

---

## Layer 2 — Hybrid Retrieval (`retrieval/hybrid_retriever.py`)

### Why hybrid
Plain vector search over all 718 chunks retrieves semantically similar text regardless of loan type or time period. A CRITICAL-tier mortgage query should not surface LOW-tier auto loan notes. Metadata pre-filtering constrains the search space before vector similarity is computed.

### Step-by-step

**1. Metadata pre-filter**
```python
candidate_indices = [i for i, m in enumerate(metadata)
                     if m.get("loan_type") == borrower["loan_type"]]
```
Filters by any combination of `loan_type`, `risk_band`, `vintage_year`. Returns indices into the FAISS metadata array.

**2. Vector similarity over filtered subset**
Candidate vectors are reconstructed from the FAISS index via `index.reconstruct(i)`. Cosine similarity is computed as a dot product (vectors are pre-normalised).

**3. Deduplication + re-ranking**
Two seen-sets prevent redundant results:
- `seen_notes` — by `note_id`: same note cannot appear twice
- `seen_texts` — by first 80 characters: catches duplicate content with different IDs

The retriever oversamples `top_k × 8` candidates before deduplication to ensure `top_k` diverse results are returned even when many candidates share the same note.

**4. Similarity threshold**
Chunks below `MIN_SIMILARITY_SCORE` (default 0.30, set in `.env`) are discarded. Set too high (e.g. 0.70) and only near-identical complaint notes pass; 0.30 allows diverse note types at the cost of some noise.

### Query enrichment
The query string is enriched with domain keywords beyond the borrower ID:
```python
f"credit risk payment history delinquency collections hardship {loan_type} borrower {borrower_id}"
```
This improves recall for underwriter and collections notes vs. a bare entity lookup.

---

## Layer 3 — ML Scoring (`generation/xgb_scorer.py`, `generation/shap_explainer.py`)

### Model
- **Type:** XGBoost binary classifier (`XGBClassifier`)
- **Target:** 1 = HIGH or CRITICAL risk, 0 = LOW or MEDIUM
- **Features (13):** fico_score, dti_ratio, ltv_ratio, payments_late_30/60/90d, num_deferrals, employment_gap_months, num_open_accounts, credit_history_yrs, annual_income, loan_amount, loan_term_months
- **Hyperparameters:** 200 estimators, max_depth=4, learning_rate=0.05, subsample=0.8

### Tier mapping
```python
if p >= 0.75: return "CRITICAL"
if p >= 0.50: return "HIGH"
if p >= 0.25: return "MEDIUM"
return "LOW"
```

### Persistence workaround
`XGBClassifier.save_model()` raises `TypeError: _estimator_type undefined` in certain XGBoost versions. Fixed by saving and loading via the raw booster:
```python
# Save
self._model.get_booster().save_model(str(MODEL_PATH))

# Load
self._booster = xgb.Booster()
self._booster.load_model(str(path))
# Score via DMatrix, not sklearn wrapper
dmatrix = xgb.DMatrix(row_df)
prob = float(self._booster.predict(dmatrix)[0])
```

### SHAP
`SHAPExplainer` wraps the booster with `shap.TreeExplainer`. Top-N features by absolute SHAP value are returned as `(feature_name, shap_value)` tuples and formatted into the LLM prompt alongside retrieved chunks.

---

## Layer 4 — LLM Generation (`generation/generator.py`, `generation/context_builder.py`)

### Context assembly (`context_builder.py`)
Three blocks are assembled in order:
1. **Structured borrower signals** — key-value table of all 13 features
2. **XGBoost signal** — predicted tier, probability, top SHAP features (if enabled)
3. **Retrieved text chunks** — ordered by similarity score, with note type and date prefix

### LLM routing
| `LLM_PROVIDER` | Client | Notes |
|---|---|---|
| `groq` | `openai.OpenAI` | `OPENAI_BASE_URL=https://api.groq.com/openai/v1` |
| `openai` | `openai.OpenAI` | Supports `response_format=json_object` |
| `ollama` | `httpx` POST | `/api/generate`, stream=False |
| `anthropic` | `anthropic.Anthropic` | `client.messages.create` |

All providers use `temperature=0.0` for deterministic outputs.

### Output schema
The system prompt and decision template instruct the LLM to respond **only** with valid JSON:
```json
{
  "risk_tier": "HIGH | MEDIUM | LOW | CRITICAL",
  "confidence": 0.0,
  "decision": "string",
  "key_signals": ["string"],
  "reasoning": "string"
}
```

`_parse_output()` strips markdown fences before JSON parsing. On parse failure, returns a sentinel object (`risk_tier: UNKNOWN`, `decision: "Parse error"`) rather than crashing.

---

## Layer 5 — Evaluation (`evaluation/metrics.py`, `evaluation/run_eval.py`)

### LLM-as-judge metrics
Three metrics scored by a second LLM call (same provider, `max_tokens=10`):

| Metric | Question asked to LLM | Weight |
|---|---|---|
| Faithfulness | Are key signals grounded in retrieved context? | 35% |
| Relevance | Are retrieved chunks relevant to the borrower query? | 30% |
| Completeness | Does output cover payment history, DTI, credit profile, recommendation? | 25% |
| Decision accuracy | `predicted_tier == analyst_risk_tier` (deterministic) | 10% |

### Results interpretation
- **Faithfulness 0.50** — with `MIN_SIMILARITY_SCORE=0.30`, some retrieved chunks are low-relevance complaint notes. Raising the threshold improves faithfulness but reduces chunk diversity.
- **Decision accuracy 0.40** — 86% of synthetic borrowers are HIGH/CRITICAL. The LLM tends to assign MEDIUM on borderline HIGH cases, reducing agreement with the analyst label.
- **Completeness 0.92** — the structured prompt template reliably produces all required output sections.

---

## UI Architecture (`app/streamlit_app.py`)

### 3-panel layout
```
st.columns([1.05, 1.35, 1.45], gap="medium")

Panel 1 — Input          Panel 2 — Retrieval       Panel 3 — Decision
─────────────────────    ──────────────────────    ───────────────────────
Risk tier filter         Structured signals        Tier card
Loan type filter         (10 fields, live)         Confidence + stats row
Borrower selector        Retrieved evidence        Structured JSON output
Borrower snapshot card   (colour-coded by          Key signal chips
top-k slider             note type, sim score,     Natural language summary
Reasoning toggle         note ID)                  Reasoning chain
XGBoost toggle                                     XGBoost signal
Generate button
```

### Theming
CSS custom properties defined in `:root` (light) and overridden in `@media (prefers-color-scheme: dark)`. No toggle required — matches OS appearance automatically.

| Token | Light | Dark |
|---|---|---|
| `--bg-base` | `#F5F3EF` | `#0C0A09` |
| `--bg-surface` | `#FFFFFF` | `#161412` |
| `--accent` | `#6366F1` | `#818CF8` |
| `--tier-crit-text` | `#B91C1C` | `#F87171` |

Each risk tier has a dedicated color set (text, background, border) for both modes.

---

## Swap Guide

| Component | Current | How to swap |
|---|---|---|
| Embeddings | Ollama `nomic-embed-text` | Set `EMBEDDING_PROVIDER=openai` in `.env` |
| Vector store | FAISS in-memory | Replace `HybridRetriever.__init__` with Pinecone/BQ client |
| LLM | Groq `llama-3.3-70b-versatile` | Set `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-sonnet-4-20250514` |
| ML model | XGBoost binary | Replace `CreditScorer` — interface is `train(df)` / `score(row) → XGBoostSignal` |
| Evaluation tracking | None | `pip install mlflow && mlflow ui` — already wired in `run_eval.py` |
