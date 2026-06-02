"""
embedder.py
-----------
Dense embedding via Ollama + BM25 sparse encoding.

Fix vs previous version:
  - _bm25_locks now use threading.RLock() (reentrant) instead of Lock().
    The deadlock occurred in embed_texts_sparse(): it acquired the lock, then
    called load_bm25_encoder() which tried to acquire the SAME lock → hung
    forever with no error. RLock allows the same thread to re-enter safely.
  - All other logic unchanged.
"""

import os
import time
import threading
import requests
from typing import List, Dict, Any, Optional

from bm25_encoder import BM25Encoder

OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL",     "qwen3-embedding:0.6b")
BM25_VOCAB_DIR      = os.getenv("BM25_VOCAB_DIR",      ".")

MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "200000"))
OLLAMA_TIMEOUT      = int(os.getenv("OLLAMA_TIMEOUT",  "300"))
OLLAMA_MAX_RETRIES  = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))


# ── Per-collection BM25 encoder registry ─────────────────────────────────────
#
# Uses RLock (reentrant lock) so the same thread can acquire the lock multiple
# times without deadlocking. This matters because embed_texts_sparse() holds
# the lock and may call load_bm25_encoder() which also tries to acquire it.

_bm25_encoders: Dict[str, BM25Encoder]    = {}
_bm25_locks:    Dict[str, threading.RLock] = {}   # RLock, not Lock
_registry_lock  = threading.Lock()


def _get_encoder_and_lock(collection: str):
    """Return (encoder, RLock) for the given collection, creating if needed."""
    with _registry_lock:
        if collection not in _bm25_locks:
            _bm25_locks[collection] = threading.RLock()   # reentrant
        lock = _bm25_locks[collection]
    encoder = _bm25_encoders.get(collection)
    return encoder, lock


def _vocab_path(collection: str) -> str:
    filename = f"bm25_vocab_{collection}.json"
    return os.path.join(BM25_VOCAB_DIR, filename)


def load_bm25_encoder(collection: str) -> BM25Encoder:
    """
    Load (or create) the BM25Encoder for this collection from its vocab file.
    Called by ingestor.py BEFORE embed_chunks() so the right vocab is in place.
    Safe to call while already holding the collection RLock (reentrant).
    """
    _, lock = _get_encoder_and_lock(collection)
    with lock:
        path = _vocab_path(collection)
        enc  = BM25Encoder.load_or_create(path)
        _bm25_encoders[collection] = enc
        print(f"[embedder] BM25 encoder loaded for collection='{collection}' path={path}")
        return enc


def save_bm25_encoder(collection: str):
    """Persist the BM25Encoder for this collection to its vocab file."""
    _, lock = _get_encoder_and_lock(collection)
    with lock:
        enc = _bm25_encoders.get(collection)
        if enc is not None:
            path = _vocab_path(collection)
            enc.save(path)
            print(f"[embedder] BM25 encoder saved for collection='{collection}' path={path}")
        else:
            print(f"[embedder] Warning: no encoder in memory for collection='{collection}', skipping save")


# ── Legacy shim ───────────────────────────────────────────────────────────────

def get_bm25_encoder() -> BM25Encoder:
    enc = _bm25_encoders.get("_global_")
    if enc is None:
        enc = load_bm25_encoder("_global_")
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# Batch builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_batches(texts: List[str]) -> List[List[int]]:
    """Group text indices into batches within MAX_CHARS_PER_BATCH."""
    batches, current_batch, current_chars = [], [], 0
    for i, text in enumerate(texts):
        n = len(text)
        if current_batch and current_chars + n > MAX_CHARS_PER_BATCH:
            batches.append(current_batch)
            current_batch, current_chars = [], 0
        current_batch.append(i)
        current_chars += n
    if current_batch:
        batches.append(current_batch)
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Ollama HTTP call — per-batch timeout + retry
# ─────────────────────────────────────────────────────────────────────────────

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Send one batch to Ollama. Retries up to OLLAMA_MAX_RETRIES times on
    transient failures (timeout, 5xx). Connection errors are fatal immediately.
    """
    url     = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": EMBEDDING_MODEL, "input": texts}

    last_exc = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            t0       = time.time()
            response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            elapsed  = time.time() - t0
            vectors  = response.json()["embeddings"]
            print(f"[embedder] Ollama batch of {len(texts)} texts → "
                  f"dim={len(vectors[0])} in {elapsed:.1f}s "
                  f"({elapsed/len(texts):.2f}s/chunk)")
            return vectors

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"[embedder] Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Is 'ollama serve' running?"
            ) from e

        except requests.exceptions.Timeout:
            last_exc = RuntimeError(
                f"[embedder] Ollama timed out after {OLLAMA_TIMEOUT}s "
                f"(attempt {attempt}/{OLLAMA_MAX_RETRIES}, batch size={len(texts)}). "
                "Consider lowering MAX_CHARS_PER_BATCH or increasing OLLAMA_TIMEOUT."
            )
            wait = 2 ** attempt
            print(f"[embedder] ⚠️  timeout on attempt {attempt}, retrying in {wait}s …")
            time.sleep(wait)

        except Exception as e:
            last_exc = RuntimeError(f"[embedder] Embedding failed: {e}")
            wait = 2 ** attempt
            print(f"[embedder] ⚠️  error on attempt {attempt}: {e}, retrying in {wait}s …")
            time.sleep(wait)

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Dense-embed a list of strings via Ollama. Returns vectors in same order."""
    if not texts:
        return []

    batches     = _build_batches(texts)
    all_vectors = [None] * len(texts)
    total_start = time.time()

    print(f"[embedder] Dense embedding {len(texts)} texts in {len(batches)} batches "
          f"(MAX_CHARS_PER_BATCH={MAX_CHARS_PER_BATCH}, OLLAMA_TIMEOUT={OLLAMA_TIMEOUT}s)")

    for batch_num, batch_indices in enumerate(batches, 1):
        batch_texts   = [texts[i] for i in batch_indices]
        batch_vectors = _embed_batch(batch_texts)
        for idx, vec in zip(batch_indices, batch_vectors):
            all_vectors[idx] = vec
        done_so_far = sum(len(b) for b in batches[:batch_num])
        print(f"[embedder] Dense batch {batch_num}/{len(batches)} done "
              f"({done_so_far}/{len(texts)} total, "
              f"elapsed={time.time()-total_start:.0f}s)")

    print(f"[embedder] Dense embedding complete in {time.time()-total_start:.1f}s")
    return all_vectors


def embed_texts_sparse(
    texts: List[str],
    collection: str = "_global_",
    add_new_tokens: bool = True,
) -> List[Dict]:
    """
    BM25-encode a list of strings for a specific collection.
    Returns list of {"indices": [...], "values": [...]} dicts.
    No Ollama involved — pure in-memory BM25.

    Previously deadlocked: held RLock then called load_bm25_encoder() which
    tried to acquire the same lock. Now safe because:
      - Lock is RLock (reentrant) — same thread can acquire multiple times
      - load_bm25_encoder() is called inside the same lock scope safely
    """
    _, lock = _get_encoder_and_lock(collection)
    with lock:
        enc = _bm25_encoders.get(collection)
        if enc is None:
            # RLock allows this re-entrant call without deadlock
            enc = load_bm25_encoder(collection)
        return enc.encode_batch(texts, add_new_tokens=add_new_tokens)


def embed_chunks(
    chunks: List[Dict[str, Any]],
    collection: str = "_global_",
) -> List[Dict[str, Any]]:
    """
    Add both 'vector' (dense) and 'sparse_vector' (BM25) to each chunk.

    Steps:
      1. Build text representations for all chunks.
      2. Fit BM25 on this corpus (IDF update) — under per-collection RLock.
      3. Dense-embed via Ollama (no lock; Ollama is external).
      4. BM25-encode — under per-collection RLock.
      5. Attach both to each chunk dict.
    """
    if not chunks:
        return chunks

    texts = [
        f"{c.get('doc_type', '')} {c.get('name', '')} {c['text']}".strip()
        for c in chunks
    ]

    _, lock = _get_encoder_and_lock(collection)

    # ── Fit BM25 (IDF update) ─────────────────────────────────────────────────
    with lock:
        enc = _bm25_encoders.get(collection)
        if enc is None:
            enc = load_bm25_encoder(collection)   # safe: RLock is reentrant
        enc.fit(texts)

    # ── Dense vectors (Ollama — no lock needed) ───────────────────────────────
    print(f"[embedder] Computing dense vectors for {len(chunks)} chunks "
          f"(collection='{collection}')...")
    dense_vectors = embed_texts(texts)

    # ── Sparse vectors (BM25) ─────────────────────────────────────────────────
    print(f"[embedder] Computing BM25 sparse vectors for {len(chunks)} chunks "
          f"(collection='{collection}')...")
    with lock:
        enc            = _bm25_encoders.get(collection)
        sparse_vectors = enc.encode_batch(texts, add_new_tokens=True)

    for chunk, dvec, svec in zip(chunks, dense_vectors, sparse_vectors):
        chunk["vector"]        = dvec
        chunk["sparse_vector"] = svec

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
    TEST_COLLECTION = "test_repo"

    print(f"Testing dense: {EMBEDDING_MODEL}")
    vecs = embed_texts(test_texts)
    print(f"Dense: {len(vecs)} vectors, dim={len(vecs[0])}")

    enc = load_bm25_encoder(TEST_COLLECTION)
    _, lock = _get_encoder_and_lock(TEST_COLLECTION)
    with lock:
        enc.fit(test_texts)
        svecs = enc.encode_batch(test_texts)
    print(f"Sparse: {len(svecs)} vectors, first has {len(svecs[0]['indices'])} non-zero terms")
    save_bm25_encoder(TEST_COLLECTION)
    print(f"Saved BM25 vocab to {_vocab_path(TEST_COLLECTION)}")
    print("✅ Both encoders work!")