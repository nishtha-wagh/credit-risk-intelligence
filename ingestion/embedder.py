"""
ingestion/embedder.py

Embeds text chunks and stores them in a FAISS index.
Supports two embedding providers — set EMBEDDING_PROVIDER in .env:

  EMBEDDING_PROVIDER=ollama   → nomic-embed-text via local Ollama (free, no API key)
  EMBEDDING_PROVIDER=openai   → OpenAI or any OpenAI-compatible API

Default: ollama (fully free)
"""

import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

from ingestion.chunker import Chunk

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
BATCH_SIZE         = 20


def embed_and_store(
    chunks: list[Chunk],
    index_path: str = "data/processed/index.faiss",
    metadata_path: str = "data/processed/metadata.json",
) -> None:
    """
    Embed all chunks and persist:
      - FAISS index  → index_path
      - Metadata     → metadata_path
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu")

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    texts    = [c.text for c in chunks]
    metadata = [c.to_dict() for c in chunks]

    print(f"[embedder] Provider : {EMBEDDING_PROVIDER}")
    print(f"[embedder] Model    : {EMBEDDING_MODEL}")
    print(f"[embedder] Chunks   : {len(texts)}")

    if EMBEDDING_PROVIDER == "ollama":
        vectors = _embed_ollama(texts)
    else:
        vectors = _embed_openai(texts)

    embed_dim = len(vectors[0])
    print(f"[embedder] Dimension: {embed_dim}")

    matrix = np.array(vectors, dtype="float32")
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-10)

    index = faiss.IndexFlatIP(embed_dim)
    index.add(matrix)

    faiss.write_index(index, index_path)
    print(f"[embedder] FAISS index saved → {index_path}  ({index.ntotal} vectors)")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[embedder] Metadata saved  → {metadata_path}")


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    """
    Call Ollama /api/embeddings endpoint.
    Run: ollama pull nomic-embed-text
    """
    client  = httpx.Client(timeout=60.0)
    vectors = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (Ollama)"):
        batch = texts[i: i + BATCH_SIZE]
        for text in batch:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])

    client.close()
    return vectors


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """
    OpenAI-compatible embeddings.
    Set OPENAI_BASE_URL in .env to point at a different provider.
    """
    from openai import OpenAI
    client  = OpenAI()
    vectors = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (OpenAI)"):
        batch    = texts[i: i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([r.embedding for r in response.data])
        time.sleep(0.05)

    return vectors
