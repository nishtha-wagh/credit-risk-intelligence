"""
ingestion/chunker.py

Splits case note text into overlapping chunks for embedding.
Attaches metadata from the parent note row for hybrid retrieval filtering.
"""

from dataclasses import dataclass, field
import re


@dataclass
class Chunk:
    chunk_id: str
    note_id: str
    borrower_id: str
    chunk_index: int
    text: str
    token_estimate: int

    # Metadata for pre-filtering (denormalised)
    loan_type: str = ""
    vintage_year: int = 0
    risk_band: str = ""
    note_type: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def chunk_notes(
    notes_df,
    chunk_size: int = 300,   # tokens (approx 4 chars each)
    overlap: int = 50,
) -> list[Chunk]:
    """
    Chunk all case notes into overlapping text segments.
    Returns a flat list of Chunk objects with metadata attached.
    """
    chunks = []
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    for _, row in notes_df.iterrows():
        text = _clean(row["note_text"])
        raw_chunks = _split(text, char_size, char_overlap)

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{row['note_id']}_c{i}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                note_id=row["note_id"],
                borrower_id=row["borrower_id"],
                chunk_index=i,
                text=chunk_text,
                token_estimate=len(chunk_text) // 4,
                loan_type=str(row.get("loan_type", "")),
                vintage_year=int(row.get("vintage_year", 0)),
                risk_band=str(row.get("risk_band", "")),
                note_type=str(row.get("note_type", "")),
            ))

    print(f"[chunker] Produced {len(chunks)} chunks from {len(notes_df)} notes")
    return chunks


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = start + size
        parts.append(text[start:end].strip())
        start += size - overlap

    return [p for p in parts if len(p) > 20]
