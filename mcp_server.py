"""
mcp_server.py
-------------
Single-file MCP server with search logic merged in.
Replaces both search_api.py and server_sse.py.

Run:
    uv run mcp_server.py

Expose:
    ngrok http 8080

Add to Claude Web Integrations:
    https://YOUR-NGROK.ngrok-free.app/mcp
"""

import os
from typing import Optional, List, Dict, Any, Set
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from embedder import embed_texts, embed_texts_sparse
from qdrant_store import search as qdrant_search, fetch_by_ids
from describer import rerank

load_dotenv(".env.claude")

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))

mcp = FastMCP("rag-search", stateless_http=True)

# warm up embedding model on start
print("[mcp_server] Warming up embedding model...")
try:
    embed_texts(["warmup"])
except Exception as e:
    print(f"[mcp_server] Warmup warning: {e}")
print("[mcp_server] Ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (copied as-is from search_api.py)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_key(r: Dict[str, Any]) -> tuple:
    stored_id = r.get("id", "")
    if stored_id:
        return (stored_id,)
    return (r.get("file_path", ""), r.get("name", ""))


def _dedup(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen:   Set[tuple]           = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = _chunk_key(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _expand(
    vector_results: List[Dict[str, Any]],
    collection:     str,
    called_by_hops: int,
    expand_calls:   bool,
) -> List[Dict[str, Any]]:
    to_fetch:     Set[str]       = set()
    fetch_reason: Dict[str, str] = {}

    for r in vector_results:
        parent_name = r.get("name", "?")
        if expand_calls:
            for cid in r.get("calls", []):
                if cid not in fetch_reason:
                    to_fetch.add(cid)
                    fetch_reason[cid] = f"calls:{parent_name}"

    if not to_fetch:
        return []

    fetched_ids: Set[str] = set(to_fetch)
    direct = fetch_by_ids(collection, list(to_fetch))

    for r in direct:
        cid = r.get("id", "")
        r["expand_reason"] = fetch_reason.get(cid, "graph")

    return list(direct)


def _search(
    query:          str,
    collection:     str,
    top_k:          int  = 5,
    branch:         str  = None,
    repo_url:       str  = None,
    expand_calls:   bool = True,
    called_by_hops: int  = 1,
    search_mode:    str  = "hybrid",
) -> Dict[str, Any]:
    """Core search logic — identical to search_endpoint() in search_api.py."""

    query_vector = None
    if search_mode in ("hybrid", "dense"):
        query_vector = embed_texts([
            f"Represent this code search query for retrieving relevant code: {query}"
        ])[0]

    sparse_vector = None
    if search_mode in ("hybrid", "sparse"):
        sparse_vectors = embed_texts_sparse(
            [query],
            collection=collection,
            add_new_tokens=False,
        )
        sparse_vector = sparse_vectors[0] if sparse_vectors else None
        idx_count = len(sparse_vector.get("indices", [])) if sparse_vector else 0
        print(f"[search] sparse indices count: {idx_count}")
        if not idx_count:
            print(f"[search] ⚠️  sparse vector empty — collection='{collection}'")

    vector_results = qdrant_search(
        collection_name=collection,
        query_vector=query_vector,
        top_k=top_k,
        filter_repo=repo_url,
        filter_branch=branch,
        sparse_vector=sparse_vector,
        search_mode=search_mode,
    )
    print(f"[search] vector results: {len(vector_results)}")

    expanded = _expand(
        vector_results,
        collection=collection,
        called_by_hops=called_by_hops,
        expand_calls=expand_calls,
    )
    print(f"[search] expanded chunks: {len(expanded)}")

    merged = _dedup(vector_results + expanded)
    print(f"[search] after dedup: {len(merged)}")

    if not merged:
        return {"results": [], "count": 0, "sources": {}}

    vector_hits   = [r for r in merged if "expand_reason" not in r]
    expanded_hits = [r for r in merged if "expand_reason" in r]
    results       = vector_hits[:top_k] + expanded_hits

    return {
        "results": results,
        "count":   len(results),
        "sources": {
            "vector":      len(vector_results),
            "expanded":    len(expanded),
            "merged":      len(merged),
            "search_mode": search_mode,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_codebase(
    query:          str,
    collection:     str,
    top_k:          int  = 5,
    branch:         str  = None,
    repo_url:       str  = None,
    expand_calls:   bool = True,
    called_by_hops: int  = 1,
    search_mode:    str  = "hybrid",
) -> str:
    """
    Search the RAG codebase index.

    WHEN TO CALL THIS TOOL:
    - User asks about code, functions, files, logic, or architecture of the indexed repo.
    - User asks a follow-up that needs NEW information not already in the conversation.

    WHEN NOT TO CALL THIS TOOL:
    - User asks to explain or elaborate on code ALREADY returned in this conversation.
    - User asks a general programming question unrelated to the codebase.
    - User is just chatting or asking meta questions.

    BEFORE CALLING:
    1. Rewrite the user's question into a short, dense, keyword-rich search query.
       Strip greetings, filler words, pronouns. Max 1-2 sentences.
       BAD:  "hey can you tell me how chunks get embedded and saved?"
       GOOD: "chunk embedding storage pipeline"

    2. Always confirm the `collection` name with the user if not already known.
       collection = the Qdrant collection = repo identifier (e.g. "abc1").
       NEVER guess it.

    ARGS:
      query          : clean rewritten search query (NOT raw user message)
      collection     : Qdrant collection name (e.g. "abc1") — always required
      top_k          : number of vector results (default 5)
      branch         : optional git branch filter
      repo_url       : optional repo URL filter
      expand_calls   : expand function call graph (default True)
      called_by_hops : depth of called_by expansion (default 1)
      search_mode    : "hybrid" | "dense" | "sparse" (default "hybrid")

    AFTER RECEIVING RESULTS:
    - Results may contain repeated or overlapping code chunks (no reranker used).
    - Read ALL results first. Understand what the query is really about.
    - Synthesize a clear answer — do not dump raw chunks at the user.
    - If two chunks look similar, use the more complete one.
    - Always cite file_path and function name when referencing code.
    """

    if not query.strip():
        return "Error: query cannot be empty."

    data = _search(
        query=query,
        collection=collection,
        top_k=top_k,
        branch=branch,
        repo_url=repo_url,
        expand_calls=expand_calls,
        called_by_hops=called_by_hops,
        search_mode=search_mode,
    )

    results = data.get("results", [])
    sources = data.get("sources", {})

    if not results:
        return "No results found. Try a different query or check the collection name."

    lines = [
        f"[Search: '{query}' in '{collection}' | "
        f"vector={sources.get('vector', 0)} "
        f"expanded={sources.get('expanded', 0)} "
        f"mode={sources.get('search_mode', '?')}]\n"
    ]

    for i, r in enumerate(results, 1):
        reason = f" [{r['expand_reason']}]" if "expand_reason" in r else ""
        score  = f" score={r['score']:.3f}" if "score" in r else ""
        lines.append(
            f"── Result {i}{reason}{score}\n"
            f"   file : {r.get('file_path', '?')}\n"
            f"   func : {r.get('name', '?')} "
            f"\n"
            f"   code :\n{r.get('text', '')}\n"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)