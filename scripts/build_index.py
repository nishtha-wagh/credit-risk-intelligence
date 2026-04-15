"""
scripts/build_index.py

One-time script: load notes → chunk → embed → save FAISS index + metadata.

Run after generate_mock_data.py:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.loader import load_case_notes
from ingestion.chunker import chunk_notes
from ingestion.embedder import embed_and_store


def main():
    print("=== Building RAG Index ===\n")

    print("Step 1/3: Loading case notes...")
    notes_df = load_case_notes("data/raw/case_notes.csv")

    print("\nStep 2/3: Chunking...")
    chunks = chunk_notes(notes_df, chunk_size=300, overlap=50)

    print("\nStep 3/3: Embedding + saving index...")
    embed_and_store(
        chunks,
        index_path="data/processed/index.faiss",
        metadata_path="data/processed/metadata.json",
    )

    print("\n✅ Index build complete. Ready to run API or Streamlit app.")


if __name__ == "__main__":
    main()
