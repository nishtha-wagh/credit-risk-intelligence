-- ============================================================
-- SCHEMA: Credit Risk RAG System
-- Compatible with BigQuery and PostgreSQL
-- ============================================================

-- Core borrower table (structured signals)
CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id         STRING NOT NULL,          -- e.g. "B-4821"
    loan_id             STRING NOT NULL,
    loan_type           STRING NOT NULL,          -- auto | mortgage | personal | student
    vintage_year        INT64,                    -- origination year
    loan_amount         FLOAT64,
    loan_term_months    INT64,

    -- Credit profile
    fico_score          INT64,                    -- 300–850
    dti_ratio           FLOAT64,                  -- debt-to-income (0.0–1.0)
    ltv_ratio           FLOAT64,                  -- loan-to-value (mortgages)
    num_open_accounts   INT64,
    credit_history_yrs  FLOAT64,

    -- Payment behaviour
    payments_on_time    INT64,
    payments_late_30d   INT64,
    payments_late_60d   INT64,
    payments_late_90d   INT64,
    num_deferrals       INT64,
    ever_defaulted      BOOL,

    -- Employment
    employment_status   STRING,                   -- employed | self-employed | unemployed | retired
    employment_gap_months INT64,                  -- months gap at origination
    annual_income       FLOAT64,

    -- Risk labels (ground truth for evaluation)
    analyst_risk_tier   STRING,                   -- LOW | MEDIUM | HIGH | CRITICAL
    in_collections      BOOL,
    charged_off         BOOL,

    -- Timestamps
    origination_date    DATE,
    last_payment_date   DATE,
    record_updated_at   TIMESTAMP
);

-- Unstructured case notes (text for RAG)
CREATE TABLE IF NOT EXISTS case_notes (
    note_id             STRING NOT NULL,
    borrower_id         STRING NOT NULL,          -- FK → borrowers.borrower_id
    loan_id             STRING,
    note_type           STRING,                   -- underwriter | collections | servicing | complaint
    note_date           DATE,
    author_role         STRING,                   -- underwriter | analyst | agent
    note_text           TEXT NOT NULL,            -- raw text for embedding

    -- Metadata for hybrid retrieval pre-filtering
    loan_type           STRING,
    vintage_year        INT64,
    risk_band           STRING,                   -- LOW | MEDIUM | HIGH | CRITICAL

    created_at          TIMESTAMP
);

-- Embedded chunks (populated by ingestion pipeline)
CREATE TABLE IF NOT EXISTS embedded_chunks (
    chunk_id            STRING NOT NULL,
    note_id             STRING NOT NULL,          -- FK → case_notes.note_id
    borrower_id         STRING NOT NULL,
    chunk_index         INT64,                    -- position within note
    chunk_text          TEXT NOT NULL,
    token_count         INT64,

    -- Metadata for filtering (denormalized for query speed)
    loan_type           STRING,
    vintage_year        INT64,
    risk_band           STRING,
    note_type           STRING,

    -- Vector (stored in FAISS; this column for reference / BQ Vector Search)
    embedding_model     STRING,                   -- e.g. text-embedding-3-small
    embedding_dim       INT64,                    -- e.g. 1536

    created_at          TIMESTAMP
);

-- Assessment log (audit trail of all RAG outputs)
CREATE TABLE IF NOT EXISTS assessment_log (
    assessment_id       STRING NOT NULL,
    borrower_id         STRING NOT NULL,
    request_timestamp   TIMESTAMP,

    -- RAG output (structured)
    risk_tier           STRING,
    confidence          FLOAT64,
    decision_text       TEXT,
    key_signals         JSON,                     -- array of signal strings
    reasoning_text      TEXT,

    -- Retrieval metadata
    retrieval_score     FLOAT64,
    sources_used        INT64,
    chunks_retrieved    JSON,                     -- array of chunk_ids used

    -- Latency
    retrieval_latency_ms  INT64,
    generation_latency_ms INT64,
    total_latency_ms      INT64,

    -- Analyst override (for feedback loop)
    analyst_override    BOOL,
    analyst_risk_tier   STRING,
    override_reason     TEXT
);
