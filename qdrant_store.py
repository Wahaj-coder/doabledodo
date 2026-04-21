"""
qdrant_store.py
---------------
All Qdrant operations in one place:
  - create_collection_if_missing()
  - upsert_chunks()
  - delete_chunks_for_files()
  - search()

Uses qdrant-client (local mode OR server mode based on .env).

Local mode  (testing):  QDRANT_MODE=local   → stores in ./qdrant_data/
Server mode (Hetzner):  QDRANT_MODE=server  → connects to QDRANT_URL

Setup:
    pip install qdrant-client

Docker local (optional, faster than in-memory):
    docker run -p 6333:6333 qdrant/qdrant
"""

import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

QDRANT_MODE      = os.getenv("QDRANT_MODE", "server")          # "local" | "server"
QDRANT_URL       = os.getenv("QDRANT_URL",  "http://localhost:6333")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", None)
QDRANT_LOCAL_DIR = os.getenv("QDRANT_LOCAL_DIR", "./qdrant_data")

# Vector dimension — must match your embedding model:
#   nomic-embed-text  → 768
#   qwen3-embedding   → 1024  (adjust if needed)
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))


# ─────────────────────────────────────────────────────────────────────────────
# Client singleton
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_MODE == "server":
            print(f"[qdrant] Connecting to server: {QDRANT_URL}")
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        else:
            print(f"[qdrant] Using local persistent storage: {QDRANT_LOCAL_DIR}")
            os.makedirs(QDRANT_LOCAL_DIR, exist_ok=True)
            _client = QdrantClient(path=QDRANT_LOCAL_DIR)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Collection management
# ─────────────────────────────────────────────────────────────────────────────

def create_collection_if_missing(collection_name: str, vector_dim: int = VECTOR_DIM):
    """Create Qdrant collection if it does not exist yet."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        print(f"[qdrant] Creating collection '{collection_name}' dim={vector_dim}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"[qdrant] Collection '{collection_name}' already exists.")


# ─────────────────────────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────────────────────────

def upsert_chunks(collection_name: str, chunks: List[Dict[str, Any]]):
    """
    Insert or update chunks in Qdrant.
    Each chunk must have a 'vector' key (added by embedder.py).
    All other keys go into payload (metadata).

    Auto-creates collection if missing.
    """
    if not chunks:
        return

    # Detect vector dim from first chunk
    dim = len(chunks[0]["vector"])
    create_collection_if_missing(collection_name, vector_dim=dim)

    client = get_client()
    points = []

    for chunk in chunks:
        vector  = chunk.pop("vector")       # extract, don't store in payload
        payload = {k: v for k, v in chunk.items()}

        # Deterministic ID from file_path + chunk_index so re-ingest = upsert not duplicate
        uid = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{collection_name}:{chunk.get('file_path','')}:{chunk.get('chunk_index', 0)}"
        ))

        points.append(PointStruct(id=uid, vector=vector, payload=payload))

    # Batch upsert (qdrant handles large lists fine)
    UPSERT_BATCH = 100
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"[qdrant] Upserted {i + len(batch)}/{len(points)}")


# ─────────────────────────────────────────────────────────────────────────────
# Delete
# ─────────────────────────────────────────────────────────────────────────────

def delete_chunks_for_files(collection_name: str, file_paths: List[str]):
    """
    Delete ALL chunks in Qdrant whose payload.file_path matches any of the given paths.
    Called before re-indexing changed files.
    """
    if not file_paths:
        return

    client = get_client()

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return   # nothing to delete

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="file_path",
                    match=MatchAny(any=file_paths),
                )
            ]
        ),
    )
    print(f"[qdrant] Deleted old chunks for {len(file_paths)} files.")


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────
def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None
def search(
    collection_name: str,
    query_vector: List[float],
    top_k: int = 5,
    filter_repo: str = None,    # optional: filter by repo_url metadata
    filter_branch: str = None,  # optional: filter by branch metadata
) -> List[Dict[str, Any]]:
    """
    Vector similarity search.
    Returns list of dicts: {score, file_path, name, doc_type, text, ...}
    """
    client = get_client()

    must_conditions = []
    if filter_repo:
        must_conditions.append(
            FieldCondition(key="repo_url", match=MatchValue(value=filter_repo))
        )
    if filter_branch:
        must_conditions.append(
            FieldCondition(key="branch", match=MatchValue(value=filter_branch))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [
        {
            "score":     r.score,
            "file_path": r.payload.get("file_path"),
            "name":      r.payload.get("name"),
            "doc_type":  r.payload.get("doc_type"),
            "language":  r.payload.get("language"),
            "text":      r.payload.get("text"),
            "repo_url":  r.payload.get("repo_url"),
            "branch":    r.payload.get("branch"),
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_COLLECTION = "test_collection"
    dim = 768

    create_collection_if_missing(TEST_COLLECTION, vector_dim=dim)

    # Insert a dummy point
    dummy_chunks = [
        {
            "vector":      [0.1] * dim,
            "file_path":   "src/hello.py",
            "chunk_index": 0,
            "doc_type":    "code",
            "language":    "python",
            "name":        "hello_world",
            "text":        "def hello_world(): print('hello')",
            "repo_url":    "https://github.com/test/repo",
            "branch":      "main",
        }
    ]
    upsert_chunks(TEST_COLLECTION, dummy_chunks)

    # Search
    results = search(TEST_COLLECTION, [0.1] * dim, top_k=1)
    print(f"[qdrant] Search result: {results}")
    print("[qdrant] ✅ Qdrant works!")
