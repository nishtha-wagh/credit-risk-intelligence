"""
ingestion/loader.py

Load and validate raw borrower data from CSV.
Returns clean DataFrames ready for chunking and embedding.
"""

from pathlib import Path
import pandas as pd


REQUIRED_BORROWER_COLS = {
    "borrower_id", "loan_id", "loan_type", "vintage_year",
    "fico_score", "dti_ratio", "analyst_risk_tier",
}

REQUIRED_NOTES_COLS = {
    "note_id", "borrower_id", "note_type", "note_text",
    "loan_type", "vintage_year", "risk_band",
}


def load_borrowers(path: str | Path = "data/raw/borrowers.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df, REQUIRED_BORROWER_COLS, "borrowers")
    df["fico_score"] = df["fico_score"].astype(int)
    df["dti_ratio"] = df["dti_ratio"].astype(float)
    df["vintage_year"] = df["vintage_year"].astype(int)
    print(f"[loader] Loaded {len(df)} borrowers from {path}")
    return df


def load_case_notes(path: str | Path = "data/raw/case_notes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df, REQUIRED_NOTES_COLS, "case_notes")
    df = df[df["note_text"].notna() & (df["note_text"].str.strip() != "")]
    print(f"[loader] Loaded {len(df)} case notes from {path}")
    return df


def _validate(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[loader] {name} missing columns: {missing}")
