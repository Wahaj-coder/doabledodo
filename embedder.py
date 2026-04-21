"""
embedder.py
-----------
Handles all Ollama embedding calls.
Auto-batches chunks — no need to pass fixed batch size.
Batch size is decided based on total token estimate so
Ollama is never overloaded.

Model: qwen3-embedding (or nomic-embed-text as fallback)
Requires Ollama running locally:
    ollama pull qwen3-embedding   (or: ollama pull nomic-embed-text)
    ollama serve
"""

import os
import requests
from typing import List, Dict, Any

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")   # change to qwen3-embedding when available

# Auto-batch tuning
# Each chunk ~400 tokens max (from chunker MAX_CHUNK_TOKENS)
# Keep total tokens per batch under ~8000 to be safe on CPU
MAX_CHARS_PER_BATCH = 32_000   # ~8000 tokens at 4 chars/token


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_batches(texts: List[str]) -> List[List[int]]:
    """
    Return list of index-lists, where each sub-list is one batch.
    Batches are built by char-count so small files get merged
    and large files don't overflow.
    """
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_chars = 0

    for i, text in enumerate(texts):
        n = len(text)
        if current_batch and current_chars + n > MAX_CHARS_PER_BATCH:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(i)
        current_chars += n

    if current_batch:
        batches.append(current_batch)

    return batches


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama /api/embed endpoint for a list of texts.
    Returns list of embedding vectors.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Ollama returns {"embeddings": [[...], [...]]}
        return data["embeddings"]

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"[embedder] Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is 'ollama serve' running?"
        )
    except Exception as e:
        raise RuntimeError(f"[embedder] Embedding failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a plain list of strings.
    Returns list of vectors in same order.
    Auto-batches internally.
    """
    if not texts:
        return []

    batches = _build_batches(texts)
    all_vectors: List[List[float]] = [None] * len(texts)

    for batch_indices in batches:
        batch_texts   = [texts[i] for i in batch_indices]
        batch_vectors = _embed_batch(batch_texts)

        for idx, vec in zip(batch_indices, batch_vectors):
            all_vectors[idx] = vec

        print(f"[embedder] Batch of {len(batch_indices)} embedded ✓")

    return all_vectors


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes chunk dicts from chunker.py, adds 'vector' key to each.
    Returns same list with vectors added in-place.

    Embedding text = chunk['text']
    (You can enrich this later with name + doc_type prefix for better retrieval)
    """
    texts = [
        f"{c.get('doc_type', '')} {c.get('name', '')} {c['text']}".strip()
        for c in chunks
    ]

    vectors = embed_texts(texts)

    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_texts = [
        "def hello_world():\n    print('hello')",
        "SELECT * FROM users WHERE active = true",
        "# README\nThis is a test repository.",
    ]
    print(f"[embedder] Testing with model: {EMBEDDING_MODEL}")
    vecs = embed_texts(test_texts)
    print(f"[embedder] Got {len(vecs)} vectors, dim={len(vecs[0])}")
    print("[embedder] ✅ Embedding works!")
