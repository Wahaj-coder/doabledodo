"""
search_api.py
-------------
Search endpoint for the RAG server.
Embed query → search Qdrant → return top-k chunks as JSON.

This is what Claude Desktop / any client calls.
It does NOT run the ingestor — read-only, fast.

Can be merged into webhook_server.py later,
but kept separate for clarity and independent testing.

Start with:
    uvicorn search_api:app --host 0.0.0.0 --port 8001

Or add to webhook_server.py and run both on port 8000.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from embedder import embed_texts
from qdrant_store import search

app = FastAPI(title="RAG Search API")


class SearchRequest(BaseModel):
    query:      str
    collection: str                  # which repo / collection to search
    top_k:      int        = 5
    branch:     Optional[str] = None # optional branch filter
    repo_url:   Optional[str] = None # optional repo filter


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """
    Embed query and return top-k matching chunks.

    Example:
        curl -X POST http://localhost:8001/search \
             -H "Content-Type: application/json" \
             -d '{"query": "how does auth work", "collection": "myrepo", "top_k": 5}'
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # Embed the query (single string, no batching needed)
    vectors = embed_texts([req.query])
    query_vector = vectors[0]

    # Search Qdrant
    results = search(
        collection_name=req.collection,
        query_vector=query_vector,
        top_k=req.top_k,
        filter_repo=req.repo_url,
        filter_branch=req.branch,
    )

    return {"results": results, "count": len(results)}


@app.get("/health")
def health():
    return {"status": "ok"}
