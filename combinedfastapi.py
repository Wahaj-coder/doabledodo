"""
server.py
---------
Combined FastAPI server.
Runs webhook + search on a single port (default 8000).

Start with:
    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /search
    POST /webhook/github
    POST /webhook/bitbucket
    POST /trigger
    GET  /health
"""

import os, hmac, hashlib, json
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from ingestor import ingest
from embedder import embed_texts
from qdrant_store import search

load_dotenv()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").encode()
DEFAULT_BRANCH  = os.getenv("DEFAULT_BRANCH", "main")

app = FastAPI(title="RAG Server")


# ─────────────────────────────────────────────────────────────────────────────
# Signature verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_github_signature(body: bytes, sig_header: str) -> bool:
    if not sig_header or not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)


# Bitbucket does NOT reliably send signatures → never block
def verify_bitbucket_signature(body: bytes, sig_header: str) -> bool:
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:      str
    collection: str
    top_k:      int            = 5
    branch:     Optional[str]  = None
    repo_url:   Optional[str]  = None


@app.post("/search")
def search_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    vectors      = embed_texts([req.query])
    query_vector = vectors[0]

    results = search(
        collection_name=req.collection,
        query_vector=query_vector,
        top_k=req.top_k,
        filter_repo=req.repo_url,
        filter_branch=req.branch,
    )

    return {"results": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# GitHub webhook
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    sig  = request.headers.get("X-Hub-Signature-256", "")

    if WEBHOOK_SECRET and not verify_github_signature(body, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event   = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(body)

    if event == "push":
        repo_url = payload["repository"]["clone_url"]
        branch   = payload["ref"].replace("refs/heads/", "")

        changed_files = list({
            f
            for commit in payload.get("commits", [])
            for f in commit.get("added", []) + commit.get("modified", [])
        })

        removed_files = list({
            f
            for commit in payload.get("commits", [])
            for f in commit.get("removed", [])
        })

        print(f"[github] push → {repo_url} branch={branch}")

        if removed_files:
            from qdrant_store import delete_chunks_for_files
            from ingestor import _repo_name
            collection = _repo_name(repo_url)
            delete_chunks_for_files(collection, removed_files)

        if changed_files:
            background_tasks.add_task(
                ingest,
                repo_url=repo_url,
                branch=branch,
                changed_files=changed_files,
            )

    return {"status": "queued"}


# ─────────────────────────────────────────────────────────────────────────────
# Bitbucket webhook
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/bitbucket")
async def bitbucket_webhook(request: Request, background_tasks: BackgroundTasks):
    body    = await request.body()
    payload = json.loads(body)
    event   = request.headers.get("X-Event-Key", "")

    if event == "repo:push":
        repo_url = payload["repository"].get("links", {}).get("html", {}).get("href")

        if not repo_url:
            repo_url = payload["repository"].get("full_name")
            if repo_url:
                repo_url = f"https://bitbucket.org/{repo_url}"

        for change in payload.get("push", {}).get("changes", []):
            branch = change.get("new", {}).get("name", DEFAULT_BRANCH)

            print(f"[bitbucket] push → {repo_url} branch={branch} (full re-index)")

            background_tasks.add_task(
                ingest,
                repo_url=repo_url,
                branch=branch,
                changed_files=None,
            )

    return {"status": "queued"}


# ─────────────────────────────────────────────────────────────────────────────
# Manual trigger
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/trigger")
async def manual_trigger(request: Request, background_tasks: BackgroundTasks):
    body     = await request.json()
    repo_url = body.get("repo_url")
    branch   = body.get("branch", "main")
    full     = body.get("full", True)

    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url required")

    background_tasks.add_task(
        ingest,
        repo_url=repo_url,
        branch=branch,
        changed_files=None if full else body.get("changed_files", []),
    )

    return {"status": "queued", "repo_url": repo_url, "branch": branch}


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}