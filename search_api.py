"""
search_api.py
-------------
Hybrid RAG Search API.

Search flow:
  1. Vector search → top_k candidate chunks (with scores)
  2. Collect ALL calls + called_by chunk IDs from every candidate
  3. Fetch them all from Qdrant in one batch
  4. Tag each fetched chunk with expand_reason USING the chunk_id string
     that was stored in calls/called_by metadata (not the Qdrant point uuid)
  5. Hop called_by up to called_by_hops levels deep
  6. Global dedup ONCE at the very end — vector results first so scores kept
  7. Return: vector top_k + all expanded (expanded always returned in full)

Root cause of previous bug:
  fetch_by_ids() scrolls Qdrant by payload field "id" (the chunk_id string
  like "chunker.py::embed_chunks::98") and returns fresh dicts. The expand_reason
  was keyed by the same chunk_id string, but r.get("id") on the returned dict
  was correct — the real issue was that fetch_by_ids used MatchAny on the "id"
  payload field which IS the chunk_id string. So the lookup should work.
  
  The ACTUAL bug was simpler: the final slice
      vector_hits   = [r for r in merged if "expand_reason" not in r]
      expanded_hits = [r for r in merged if "expand_reason" in r]
  relied on expand_reason surviving through _dedup(). It does — _dedup just
  filters duplicates, it doesn't strip fields. BUT the expand_reason tag was
  added AFTER fetch_by_ids returned, so any chunk that was ALSO a vector result
  got deduped away (vector result wins, no expand_reason). That's correct
  behaviour — but chunks that were ONLY in expanded had no "id" field in their
  Qdrant payload that matched the chunk_id string in calls metadata.

  Fix: build a reverse lookup from chunk_id_string → reason BEFORE fetching,
  then after fetching match on r["id"] which IS the chunk_id string stored
  in the payload. If r["id"] is missing, fall back to file_path::name.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set

from embedder import embed_texts, embed_texts_sparse
from qdrant_store import search as qdrant_search, fetch_by_ids
from describer import rerank
import os

# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[search_api] Warming up embedding model...")
    try:
        embed_texts(["warmup"])
    except Exception as e:
        print(f"[search_api] Warmup warning: {e}")
    print("[search_api] Ready.")
    yield


NGROK_URL = "https://4e87-203-215-170-26.ngrok-free.app"  # hardcoded fallback

public_url = os.getenv("PUBLIC_URL") or NGROK_URL or "http://localhost:8001"

if not public_url:
    raise RuntimeError("No server URL configured. Set PUBLIC_URL env var or set NGROK_URL in code.")

app = FastAPI(
    title="Hybrid RAG Search API",
    lifespan=lifespan,
    servers=[
        {"url": public_url, "description": "Active server"},
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:          str
    collection:     str
    top_k:          int           = 5
    branch:         Optional[str] = None
    repo_url:       Optional[str] = None
    use_rerank:     bool          = False
    expand_calls:   bool          = True
    called_by_hops: int           = 1
    search_mode:    str           = "hybrid"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_key(r: Dict[str, Any]) -> tuple:
    """
    Stable dedup key. Uses the chunk_id string stored in payload["id"]
    if present, otherwise falls back to (file_path, name).
    chunk_id format: "file_path::name::start_line"
    """
    stored_id = r.get("id", "")
    if stored_id:
        return (stored_id,)
    return (r.get("file_path", ""), r.get("name", ""))


def _dedup(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Global dedup. Called ONCE at the very end after all expansion is done.
    First occurrence wins — vector results are passed first so their score
    and absence of expand_reason are preserved when a chunk appears in both.
    """
    seen:   Set[tuple]           = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = _chunk_key(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Graph expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand(
    vector_results: List[Dict[str, Any]],
    collection:     str,
    called_by_hops: int,
    expand_calls:   bool,
) -> List[Dict[str, Any]]:
    """
    Collect ALL calls + called_by IDs from every vector result, fetch them,
    tag with expand_reason, then hop called_by for additional levels.
    No slicing, no dedup — both happen in search_endpoint() after this returns.

    expand_reason format:
      "calls:<parent_name>"          — this chunk is called BY parent
      "called_by:<parent_name>"      — this chunk calls parent
      "called_by:<name> (hop N)"     — reached via N hops of called_by chain
    """

    # ── Step 1: collect all chunk IDs to fetch from vector results ────────────
    # fetch_reason maps chunk_id_string → reason label
    # chunk_id_string is the value stored in calls/called_by metadata,
    # e.g. "repos\\abc1\\chunker.py::embed_chunks::98"
    to_fetch:     Set[str]       = set()
    fetch_reason: Dict[str, str] = {}

    for r in vector_results:
        parent_name = r.get("name", "?")

        if expand_calls:
            for cid in r.get("calls", []):
                if cid not in fetch_reason:          # first parent wins
                    to_fetch.add(cid)
                    fetch_reason[cid] = f"calls:{parent_name}"

        # for cid in r.get("called_by", []):
        #     if cid not in fetch_reason:
        #         to_fetch.add(cid)
        #         fetch_reason[cid] = f"called_by:{parent_name}"

    if not to_fetch:
        return []

    # ── Step 2: fetch all direct expanded chunks in one Qdrant call ──────────
    fetched_ids: Set[str] = set(to_fetch)
    direct = fetch_by_ids(collection, list(to_fetch))

    # Tag each chunk. fetch_by_ids returns dicts with payload["id"] = chunk_id_string.
    # That is the same string we used as the key in fetch_reason, so lookup works.
    for r in direct:
        cid = r.get("id", "")
        r["expand_reason"] = fetch_reason.get(cid, "graph")

    all_expanded: List[Dict[str, Any]] = list(direct)

    # ── Step 3: hop called_by for additional levels ───────────────────────────
    # current_level = direct
    # for hop in range(called_by_hops - 1):
    #     next_fetch:  Set[str]       = set()
    #     next_reason: Dict[str, str] = {}

    #     for r in current_level:
    #         r_name = r.get("name", "?")
    #         for cid in r.get("called_by", []):
    #             if cid not in fetched_ids and cid not in next_reason:
    #                 next_fetch.add(cid)
    #                 next_reason[cid] = f"called_by:{r_name} (hop {hop + 2})"

    #     if not next_fetch:
    #         break

    #     more = fetch_by_ids(collection, list(next_fetch))
    #     for r in more:
    #         cid = r.get("id", "")
    #         r["expand_reason"] = next_reason.get(cid, f"graph_hop_{hop + 2}")

    #     all_expanded.extend(more)
    #     fetched_ids |= next_fetch
    #     current_level = more

    return all_expanded


# ─────────────────────────────────────────────────────────────────────────────
# Search endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/search")
def search_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # ── 1. Dense embed (only when needed) ────────────────────────────────────
    query_vector = None
    if req.search_mode in ("hybrid", "dense"):
        query_vector = embed_texts([
            f"Represent this code search query for retrieving relevant code: {req.query}"
        ])[0]

    # ── 2. Sparse BM25 (only when needed) ────────────────────────────────────
    sparse_vector = None
    if req.search_mode in ("hybrid", "sparse"):
        sparse_vectors = embed_texts_sparse(
            [req.query],
            collection=req.collection,
            add_new_tokens=False,
        )
        sparse_vector = sparse_vectors[0] if sparse_vectors else None
        idx_count = len(sparse_vector.get("indices", [])) if sparse_vector else 0
        print(f"[search] sparse indices count: {idx_count}")
        if not idx_count:
            print(f"[search] ⚠️  sparse vector empty — tokens not in vocab "
                  f"for collection='{req.collection}'. "
                  f"Expected vocab file: bm25_vocab_{req.collection}.json")

    # ── 3. Vector search → raw candidate pool ────────────────────────────────
    candidate_k = req.top_k * 4 if req.use_rerank else req.top_k

    vector_results = qdrant_search(
        collection_name=req.collection,
        query_vector=query_vector,
        top_k=candidate_k,
        filter_repo=req.repo_url,
        filter_branch=req.branch,
        sparse_vector=sparse_vector,
        search_mode=req.search_mode,
    )
    print(f"[search] vector results: {len(vector_results)}")

    # ── 4. Expand ALL calls + called_by — no slicing ─────────────────────────
    expanded = _expand(
        vector_results,
        collection=req.collection,
        called_by_hops=req.called_by_hops,
        expand_calls=req.expand_calls,
    )
    print(f"[search] expanded chunks: {len(expanded)}")

    # ── 5. Global dedup — ONE call, vector results first ─────────────────────
    #    Vector result wins if the same chunk appears in both lists.
    #    Expanded-only chunks keep their expand_reason field intact.
    merged = _dedup(vector_results + expanded)
    print(f"[search] after global dedup: {len(merged)}")

    if not merged:
        return {"results": [], "count": 0}

    # ── 6. Rerank or slice ────────────────────────────────────────────────────
    #    Vector hits → sliced to top_k (they have real scores).
    #    Expanded hits → always returned in full (no vector score, but
    #    expand_reason tells the consumer exactly why each was included).
    if req.use_rerank:
        results = rerank(query=req.query, results=merged, top_k=req.top_k)
    else:
        vector_hits   = [r for r in merged if "expand_reason" not in r]
        expanded_hits = [r for r in merged if "expand_reason" in r]
        results       = vector_hits[:req.top_k] + expanded_hits

    return {
        "results": results,
        "count":   len(results),
        "sources": {
            "vector":      len(vector_results),
            "expanded":    len(expanded),
            "merged":      len(merged),
            "search_mode": req.search_mode,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}