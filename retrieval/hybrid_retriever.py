"""
retrieval/hybrid_retriever.py

Implements hybrid retrieval:
  1. Metadata pre-filter  — narrows the candidate pool using structured fields
  2. Vector search        — ranks filtered candidates by semantic similarity
  3. Re-ranking           — returns top-k results with similarity scores

Embedding provider matches whatever was used at index-build time:
  EMBEDDING_PROVIDER=ollama  → nomic-embed-text via local Ollama (free)
  EMBEDDING_PROVIDER=openai  → OpenAI-compatible API
"""

import json
import os
from dataclasses import dataclass

import httpx
import numpy as np

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K              = int(os.getenv("TOP_K_RESULTS", 5))
MIN_SCORE          = float(os.getenv("MIN_SIMILARITY_SCORE", 0.70))


@dataclass
class RetrievedChunk:
    chunk_id: str
    borrower_id: str
    note_id: str
    note_type: str
    text: str
    similarity_score: float
    metadata: dict


class HybridRetriever:
    """
    Load a FAISS index + metadata once, then query with metadata filters.
    """

    def __init__(
        self,
        index_path: str = "data/processed/index.faiss",
        metadata_path: str = "data/processed/metadata.json",
    ):
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        self.faiss = faiss
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata: list[dict] = json.load(f)

        print(f"[retriever] Loaded index: {self.index.ntotal} vectors")
        print(f"[retriever] Embed provider: {EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}")

    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> list[RetrievedChunk]:
        candidate_indices = self._filter(filters)
        if not candidate_indices:
            return []

        query_vec       = self._embed(query)
        candidate_meta  = [self.metadata[i] for i in candidate_indices]
        candidate_vecs  = self._fetch_vectors(candidate_indices)
        scores          = (candidate_vecs @ query_vec).tolist()

        ranked = sorted(
            zip(scores, candidate_indices, candidate_meta),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        seen_notes = set()
        seen_texts = set()
        for score, idx, meta in ranked[:top_k * 8]:
            if score < min_score:
                break
            note_id  = meta.get("note_id", meta.get("chunk_id"))
            text_sig = meta.get("text", "")[:80]
            if note_id in seen_notes or text_sig in seen_texts:
                continue
            seen_notes.add(note_id)
            seen_texts.add(text_sig)
            results.append(RetrievedChunk(
                chunk_id=meta["chunk_id"],
                borrower_id=meta["borrower_id"],
                note_id=meta["note_id"],
                note_type=meta.get("note_type", ""),
                text=meta["text"],
                similarity_score=round(float(score), 4),
                metadata=meta,
            ))
            if len(results) >= top_k:
                break

        return results

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _filter(self, filters: dict | None) -> list[int]:
        if not filters:
            return list(range(len(self.metadata)))
        return [i for i, m in enumerate(self.metadata) if self._passes(m, filters)]

    @staticmethod
    def _passes(meta: dict, filters: dict) -> bool:
        for key, value in filters.items():
            mv = meta.get(key)
            if isinstance(value, list):
                if mv not in value:
                    return False
            else:
                if mv != value:
                    return False
        return True

    def _embed(self, text: str) -> np.ndarray:
        if EMBEDDING_PROVIDER == "ollama":
            vec = _embed_ollama_single(text)
        else:
            vec = _embed_openai_single(text)
        arr  = np.array(vec, dtype="float32")
        norm = np.linalg.norm(arr)
        return arr / max(norm, 1e-10)

    def _fetch_vectors(self, indices: list[int]) -> np.ndarray:
        dim  = self.index.d
        vecs = np.zeros((len(indices), dim), dtype="float32")
        for j, i in enumerate(indices):
            self.index.reconstruct(i, vecs[j])
        return vecs


# ---------------------------------------------------------------------------
# Embedding helpers (module-level, reusable)
# ---------------------------------------------------------------------------

def _embed_ollama_single(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _embed_openai_single(text: str) -> list[float]:
    from openai import OpenAI
    client   = OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding
