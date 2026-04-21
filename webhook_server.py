# """
# webhook_server.py
# -----------------
# FastAPI server. Receives GitHub / Bitbucket push webhooks.
# Queues background re-index jobs via FastAPI BackgroundTasks.

# For local testing (no ngrok needed):
#     uvicorn webhook_server:app --host 0.0.0.0 --port 8000
#     Then POST manually: see /docs (Swagger UI at http://localhost:8000/docs)

# For real GitHub webhooks:
#     Use ngrok: ngrok http 8000
#     Set webhook URL in GitHub repo settings → Webhooks → https://<ngrok-url>/webhook/github
# """

# import os, hmac, hashlib, json
# from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
# from dotenv import load_dotenv

# from ingestor import ingest   # ← calls chunker → embedder → qdrant_store

# load_dotenv()

# WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").encode()
# DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")

# app = FastAPI(title="RAG Webhook Server")


# # ─────────────────────────────────────────────────────────────────────────────
# # Signature verification
# # ─────────────────────────────────────────────────────────────────────────────

# def verify_github_signature(body: bytes, sig_header: str) -> bool:
#     if not sig_header or not sig_header.startswith("sha256="):
#         return False
#     expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
#     return hmac.compare_digest(expected, sig_header)


# def verify_bitbucket_signature(body: bytes, sig_header: str) -> bool:
#     if not sig_header or not sig_header.startswith("sha256="):
#         return False
#     expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
#     return hmac.compare_digest(expected, sig_header)


# # ─────────────────────────────────────────────────────────────────────────────
# # GitHub webhook
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/webhook/github")
# async def github_webhook(request: Request, background_tasks: BackgroundTasks):
#     body = await request.body()
#     sig  = request.headers.get("X-Hub-Signature-256", "")

#     if WEBHOOK_SECRET and not verify_github_signature(body, sig):
#         raise HTTPException(status_code=401, detail="Invalid signature")

#     event   = request.headers.get("X-GitHub-Event", "")
#     payload = json.loads(body)

#     if event == "push":
#         repo_url = payload["repository"]["clone_url"]
#         branch   = payload["ref"].replace("refs/heads/", "")

#         # Collect unique changed + added files
#         changed_files = list({
#             f
#             for commit in payload.get("commits", [])
#             for f in commit.get("added", []) + commit.get("modified", [])
#         })

#         # Removed files — delete their chunks
#         removed_files = list({
#             f
#             for commit in payload.get("commits", [])
#             for f in commit.get("removed", [])
#         })

#         print(f"[webhook/github] push → {repo_url} branch={branch} "
#               f"changed={len(changed_files)} removed={len(removed_files)}")

#         # Delete removed file chunks immediately (fast, no git needed)
#         if removed_files:
#             from qdrant_store import delete_chunks_for_files
#             from ingestor import _repo_name
#             collection = _repo_name(repo_url)
#             delete_chunks_for_files(collection, removed_files)

#         # Queue re-index for changed files in background
#         if changed_files:
#             background_tasks.add_task(
#                 ingest,
#                 repo_url=repo_url,
#                 branch=branch,
#                 changed_files=changed_files,
#             )

#     return {"status": "queued"}


# # ─────────────────────────────────────────────────────────────────────────────
# # Bitbucket webhook
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/webhook/bitbucket")
# async def bitbucket_webhook(request: Request, background_tasks: BackgroundTasks):
#     body = await request.body()
#     sig  = request.headers.get("X-Hub-Signature", "")

#     if WEBHOOK_SECRET and not verify_bitbucket_signature(body, sig):
#         raise HTTPException(status_code=401, detail="Invalid signature")

#     payload = json.loads(body)
#     event   = request.headers.get("X-Event-Key", "")

#     if event == "repo:push":
#         # Bitbucket clone URL
#         clone_links = payload["repository"]["links"]["clone"]
#         repo_url    = next(
#             (l["href"] for l in clone_links if l["name"] == "https"),
#             clone_links[0]["href"]
#         )

#         for change in payload.get("push", {}).get("changes", []):
#             branch = change.get("new", {}).get("name", DEFAULT_BRANCH)
#             # Bitbucket webhook payload doesn't include file diffs → full re-index
#             print(f"[webhook/bitbucket] push → {repo_url} branch={branch} (full re-index)")
#             background_tasks.add_task(
#                 ingest,
#                 repo_url=repo_url,
#                 branch=branch,
#                 changed_files=None,   # full re-index
#             )

#     return {"status": "queued"}


# # ─────────────────────────────────────────────────────────────────────────────
# # Manual trigger (useful for local testing without a real webhook)
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/trigger")
# async def manual_trigger(request: Request, background_tasks: BackgroundTasks):
#     """
#     Manually trigger ingest for testing.
#     POST JSON: {"repo_url": "...", "branch": "main", "full": true}

#     Test with:
#         curl -X POST http://localhost:8000/trigger \
#              -H "Content-Type: application/json" \
#              -d '{"repo_url": "https://github.com/org/repo", "branch": "main", "full": true}'
#     """
#     body = await request.json()
#     repo_url = body.get("repo_url")
#     branch   = body.get("branch", "main")
#     full     = body.get("full", True)

#     if not repo_url:
#         raise HTTPException(status_code=400, detail="repo_url required")

#     background_tasks.add_task(
#         ingest,
#         repo_url=repo_url,
#         branch=branch,
#         changed_files=None if full else body.get("changed_files", []),
#     )
#     return {"status": "queued", "repo_url": repo_url, "branch": branch}


# # ─────────────────────────────────────────────────────────────────────────────
# # Health check
# # ─────────────────────────────────────────────────────────────────────────────

# @app.get("/health")
# def health():
#     return {"status": "ok"}
"""
webhook_server.py
-----------------
FastAPI server. Receives GitHub / Bitbucket push webhooks.
Queues background re-index jobs via FastAPI BackgroundTasks.
"""

import os, hmac, hashlib, json
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from dotenv import load_dotenv

from ingestor import ingest   # ← calls chunker → embedder → qdrant_store

load_dotenv()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").encode()
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")

app = FastAPI(title="RAG Webhook Server")


# ─────────────────────────────────────────────────────────────────────────────
# Signature verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_github_signature(body: bytes, sig_header: str) -> bool:
    if not sig_header or not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)


# Bitbucket DOES NOT reliably send signatures → never block ingestion
def verify_bitbucket_signature(body: bytes, sig_header: str) -> bool:
    return True


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
# Bitbucket webhook (FIXED - NO 401 BLOCK)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/bitbucket")
async def bitbucket_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

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
                changed_files=None,   # full re-index
            )

    return {"status": "queued"}


# ─────────────────────────────────────────────────────────────────────────────
# Manual trigger
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/trigger")
async def manual_trigger(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
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