"""
qdrant_store.py
---------------
All Qdrant operations — now with hybrid (dense + sparse BM25) support.

Each point stores:
  vector          — dense embedding (cosine)
  sparse_vector   — BM25 sparse vector (dot-product)

search() supports three modes via `search_mode`:
  "dense"   — vector only (original behaviour)
  "sparse"  — BM25 keyword only
  "hybrid"  — both via Qdrant's query API with RRF fusion (default)
"""

import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Prefetch,
    FusionQuery,
    Fusion,
)

QDRANT_MODE      = os.getenv("QDRANT_MODE", "server")
QDRANT_URL       = os.getenv("QDRANT_URL",  "http://localhost:6333")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", None) or None
QDRANT_LOCAL_DIR = os.getenv("QDRANT_LOCAL_DIR", "./qdrant_data")
VECTOR_DIM       = int(os.getenv("VECTOR_DIM", "1024"))

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_MODE == "server":
            print(f"[qdrant] Connecting to server: {QDRANT_URL}")
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        else:
            print(f"[qdrant] Using local storage: {QDRANT_LOCAL_DIR}")
            os.makedirs(QDRANT_LOCAL_DIR, exist_ok=True)
            _client = QdrantClient(path=QDRANT_LOCAL_DIR)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Collection management
# ─────────────────────────────────────────────────────────────────────────────

def create_collection_if_missing(collection_name: str, vector_dim: int = VECTOR_DIM):
    """Create collection with both dense and sparse vector configs."""
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        print(f"[qdrant] Creating hybrid collection '{collection_name}' dim={vector_dim}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
    else:
        print(f"[qdrant] Collection '{collection_name}' already exists.")


# ─────────────────────────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────────────────────────

def upsert_chunks(collection_name: str, chunks: List[Dict[str, Any]]):
    """
    Upsert chunks with both dense vector and sparse BM25 vector.
    Each chunk must have:
      'vector'        — dense embedding list[float]
      'sparse_vector' — {"indices": [...], "values": [...]}  (from BM25Encoder)
    """
    if not chunks:
        return

    dim = len(chunks[0]["vector"])
    create_collection_if_missing(collection_name, vector_dim=dim)
    client = get_client()
    points = []

    for chunk in chunks:
        dense_vec  = chunk.pop("vector")
        sparse_vec = chunk.pop("sparse_vector", {"indices": [], "values": []})
        payload    = {k: v for k, v in chunk.items()}

        uid = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{collection_name}:{chunk.get('file_path','')}:{chunk.get('name','')}:{chunk.get('start_line', 0)}"
        ))

        point = PointStruct(
            id=uid,
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse_vec.get("indices", []),
                    values=sparse_vec.get("values", []),
                ),
            },
            payload=payload,
        )
        points.append(point)

    UPSERT_BATCH = 100
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i: i + UPSERT_BATCH]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"[qdrant] Upserted {i + len(batch)}/{len(points)}")


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search
# ─────────────────────────────────────────────────────────────────────────────

def search(
    collection_name: str,
    query_vector: Optional[List[float]],
    top_k: int = 5,
    filter_repo: str = None,
    filter_branch: str = None,
    sparse_vector: Optional[Dict] = None,
    search_mode: str = "hybrid",          # "dense" | "sparse" | "hybrid"
) -> List[Dict[str, Any]]:
    """
    Hybrid search using Qdrant's native RRF fusion.

    search_mode:
      "dense"  — cosine similarity on dense vectors only
      "sparse" — BM25 dot-product on sparse vectors only
      "hybrid" — prefetch both, fuse with RRF (best results)

    query_vector is Optional — not required for sparse-only mode.
    sparse_vector is Optional — not required for dense-only mode.
    """
    client = get_client()

    must_conditions = []
    if filter_repo:
        must_conditions.append(FieldCondition(key="repo_url", match=MatchValue(value=filter_repo)))
    if filter_branch:
        must_conditions.append(FieldCondition(key="branch", match=MatchValue(value=filter_branch)))
    query_filter = Filter(must=must_conditions) if must_conditions else None

    def _payload(r) -> Dict:
        p = r.payload
        return {
            "score":       r.score,
            "file_path":   p.get("file_path"),
            "name":        p.get("name"),
            "doc_type":    p.get("doc_type"),
            "language":    p.get("language"),
            "text":        p.get("text"),
            "description": p.get("description", ""),
            "repo_url":    p.get("repo_url"),
            "branch":      p.get("branch"),
            "calls":       p.get("calls", []),
            "called_by":   p.get("called_by", []),
        }

    # ── Dense only ────────────────────────────────────────────────────────────
    if search_mode == "dense":
        if query_vector is None:
            raise ValueError("[qdrant] query_vector is required for dense search mode")
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [_payload(r) for r in results]

    # ── Sparse only ───────────────────────────────────────────────────────────
    if search_mode == "sparse":
        if not sparse_vector or not sparse_vector.get("indices"):
            raise ValueError(
                "[qdrant] sparse_vector is empty — query tokens not found in vocab. "
                "Ensure the collection BM25 vocab is built and the query contains known tokens."
            )
        sv = SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"],
        )
        results = client.query_points(
            collection_name=collection_name,
            query=sv,
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [_payload(r) for r in results]

    # ── Hybrid (RRF) ──────────────────────────────────────────────────────────
    if query_vector is None:
        raise ValueError("[qdrant] query_vector is required for hybrid search mode")

    if not sparse_vector or not sparse_vector.get("indices"):
        # Sparse leg is unusable — degrade gracefully to dense only
        print("[qdrant] ⚠️  sparse_vector empty in hybrid mode — falling back to dense only")
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [_payload(r) for r in results]

    sv = SparseVector(
        indices=sparse_vector["indices"],
        values=sparse_vector["values"],
    )
    prefetch_k = top_k * 3  # cast a wider net before fusion
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_k,
                filter=query_filter,
            ),
            Prefetch(
                query=sv,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_k,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points
    return [_payload(r) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# fetch_by_ids
# ─────────────────────────────────────────────────────────────────────────────

def fetch_by_ids(collection_name: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []
    client  = get_client()
    results = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="id", match=MatchAny(any=chunk_ids))]),
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=False,
    )[0]
    return [
        {
            "score":       0.0,
            "file_path":   r.payload.get("file_path"),
            "name":        r.payload.get("name"),
            "doc_type":    r.payload.get("doc_type"),
            "language":    r.payload.get("language"),
            "text":        r.payload.get("text"),
            "description": r.payload.get("description", ""),
            "repo_url":    r.payload.get("repo_url"),
            "branch":      r.payload.get("branch"),
            "calls":       r.payload.get("calls", []),
            "called_by":   r.payload.get("called_by", []),
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────────────
# get_name_map
# ─────────────────────────────────────────────────────────────────────────────

def get_name_map(collection_name: str) -> Dict[str, str]:
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return {}
    name_map: Dict[str, str] = {}
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=1000, offset=offset,
            with_payload=["id", "name"], with_vectors=False,
        )
        for r in results:
            chunk_id = r.payload.get("id", "")
            name     = r.payload.get("name", "")
            if chunk_id and name:
                name_map[name]                = chunk_id
                name_map[name.split(".")[-1]] = chunk_id
        if offset is None:
            break
    print(f"[qdrant] Built name map: {len(name_map)} entries")
    return name_map


# ─────────────────────────────────────────────────────────────────────────────
# delete_chunks_for_files
# ─────────────────────────────────────────────────────────────────────────────

def delete_chunks_for_files(collection_name: str, file_paths: List[str]):
    if not file_paths:
        return
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="file_path", match=MatchAny(any=file_paths))]
        ),
    )
    print(f"[qdrant] Deleted old chunks for {len(file_paths)} files.")


# ─────────────────────────────────────────────────────────────────────────────
# close_client
# ─────────────────────────────────────────────────────────────────────────────

def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None