"""
webhook_server.py
-----------------
FastAPI server. Receives GitHub / Bitbucket push webhooks.

Design (v3):
  - done-flag tracking: last_commits.json stores { repo: { sha, done } }.
    On restart, only repos with done=false are retried (crash recovery).
    No blind auto-sync of every repo every restart.
  - First-ever webhook for a repo (no entry in last_commits.json) forces a
    full re-ingest regardless of before_sha in the payload — guarantees
    Qdrant is clean before incremental runs begin.
  - No double-queuing: only crashed (done=false) repos are re-queued on startup.
  - Per-batch Ollama timeout (300s) in embedder — if a batch hangs, exception
    propagates, done stays false, retried on next webhook or restart.
  - Job coalescing: worker drains queue before running ingest.
  - Per-repo BM25 vocab files — no shared global file.
"""

import os, hmac, hashlib, json, threading, time, shutil, subprocess, re
from queue import Queue, Empty
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

from ingestor import ingest, _repo_name

load_dotenv()

WEBHOOK_SECRET      = os.getenv("WEBHOOK_SECRET", "").encode()
DEFAULT_BRANCH      = os.getenv("DEFAULT_BRANCH", "main")
JOBS_DIR            = Path(os.getenv("JOBS_DIR", "./jobs"))
LOCAL_REPOS_DIR     = Path(os.getenv("LOCAL_REPOS_DIR", "./repos"))
WORKER_IDLE_TIMEOUT = int(os.getenv("WORKER_IDLE_TIMEOUT", "1800"))
COMMIT_STORE        = Path(os.getenv("COMMIT_STORE", "./last_commits.json"))

JOBS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Commit / done-flag store
#
# Format: { "<repo>": { "sha": "<last_successful_sha>", "done": true|false } }
#   done=true  → last ingest completed; wait for next webhook
#   done=false → server crashed mid-ingest; retry on next startup
# ─────────────────────────────────────────────────────────────────────────────

_commit_store_lock = threading.Lock()


def _load_commit_store() -> dict:
    with _commit_store_lock:
        try:
            return json.loads(COMMIT_STORE.read_text()) if COMMIT_STORE.exists() else {}
        except Exception:
            return {}


def _mark_ingest_started(repo: str):
    """Write done=false BEFORE starting ingest. Acts as crash detector."""
    with _commit_store_lock:
        try:
            store = json.loads(COMMIT_STORE.read_text()) if COMMIT_STORE.exists() else {}
        except Exception:
            store = {}
        existing_sha = store.get(repo, {}).get("sha")
        store[repo] = {"sha": existing_sha, "done": False}
        COMMIT_STORE.write_text(json.dumps(store, indent=2))
        print(f"[webhook] marked ingest started for repo={repo} (done=false)")


def _mark_ingest_done(repo: str, sha: str):
    """Write done=true and new SHA AFTER successful ingest."""
    with _commit_store_lock:
        try:
            store = json.loads(COMMIT_STORE.read_text()) if COMMIT_STORE.exists() else {}
        except Exception:
            store = {}
        store[repo] = {"sha": sha, "done": True}
        COMMIT_STORE.write_text(json.dumps(store, indent=2))
        print(f"[webhook] marked ingest done for repo={repo} sha={sha[:8]}")


def _get_last_sha(repo: str):
    store = _load_commit_store()
    return store.get(repo, {}).get("sha")


def _repo_is_known(repo: str) -> bool:
    """True only if this repo has at least one prior SUCCESSFUL ingest (sha is set)."""
    store = _load_commit_store()
    entry = store.get(repo)
    return entry is not None and entry.get("sha") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Startup lifespan — only retry crashed (done=false) repos
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    store   = _load_commit_store()
    crashed = {repo: entry for repo, entry in store.items()
               if not entry.get("done", True)}

    if crashed:
        print(f"[webhook] Found {len(crashed)} crashed ingest(s), re-queuing: {set(crashed)}")
    else:
        print("[webhook] All repos healthy (done=true). Waiting for webhooks.")

    for repo, entry in crashed.items():
        repo_path = LOCAL_REPOS_DIR / repo
        if not repo_path.exists():
            print(f"[webhook] Skipping crashed repo={repo}: local clone not found")
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
                capture_output=True, text=True, check=True,
            )
            repo_url   = re.sub(r"https://[^@]+@", "https://", result.stdout.strip())
            before_sha = entry.get("sha")  # diff from last completed SHA → HEAD
            _enqueue(repo, dict(
                repo_url=repo_url,
                branch=DEFAULT_BRANCH,
                changed_files=None,
                before_sha=before_sha,
                token=None,
            ))
            print(f"[webhook] crash-recovery queued repo={repo} "
                  f"from={before_sha[:8] if before_sha else 'full'}")
        except Exception as e:
            print(f"[webhook] could not recover crashed repo={repo}: {e}")

    yield


app = FastAPI(title="RAG Webhook Server", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Per-repo state
# ─────────────────────────────────────────────────────────────────────────────

class _RepoState:
    def __init__(self):
        self.queue  = Queue()
        self.lock   = threading.Lock()
        self.thread = None


_repos:         dict = {}
_registry_lock = threading.Lock()


def _get_state(repo: str) -> _RepoState:
    with _registry_lock:
        if repo not in _repos:
            state        = _RepoState()
            _repos[repo] = state
            state.thread = threading.Thread(target=_worker, args=(repo,), daemon=True)
            state.thread.start()
        return _repos[repo]


# ─────────────────────────────────────────────────────────────────────────────
# Job merge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _drain_queue(state: _RepoState) -> list:
    jobs = []
    while True:
        try:
            jobs.append(state.queue.get_nowait())
        except Empty:
            break
    return jobs


def _merge_jobs(jobs: list) -> dict:
    if len(jobs) == 1:
        return jobs[0]

    base = jobs[-1].copy()

    full_jobs = [j for j in jobs
                 if j.get("changed_files") is None and not j.get("before_sha")]
    if full_jobs:
        base["changed_files"] = None
        base.pop("before_sha", None)
        base.pop("after_sha", None)
        print(f"[webhook] merged {len(jobs)} jobs → full re-ingest")
        return base

    all_files: list = []
    force_full = False
    for j in jobs:
        cf = j.get("changed_files")
        if cf is not None:
            all_files.extend(cf)
        elif j.get("before_sha"):
            force_full = True

    if force_full:
        base["changed_files"] = None
        base.pop("before_sha", None)
        base.pop("after_sha", None)
        print(f"[webhook] merged {len(jobs)} jobs → full re-ingest (SHA-range present)")
        return base

    seen, merged_files = set(), []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            merged_files.append(f)

    base["changed_files"] = merged_files if merged_files else None
    base.pop("before_sha", None)
    base.pop("after_sha", None)
    print(f"[webhook] merged {len(jobs)} jobs → {len(merged_files)} changed files")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(repo: str):
    while True:
        state = _repos.get(repo)
        if state is None:
            break

        try:
            first_job = state.queue.get(timeout=WORKER_IDLE_TIMEOUT)
        except Empty:
            with _registry_lock:
                if repo in _repos and _repos[repo].queue.empty():
                    del _repos[repo]
                    print(f"[webhook] worker idle timeout — cleaned up repo={repo}")
            break

        extra_jobs = _drain_queue(state)
        all_jobs   = [first_job] + extra_jobs
        if extra_jobs:
            print(f"[webhook] coalescing {len(all_jobs)} jobs for repo={repo}")

        merged_job = _merge_jobs(all_jobs)

        # If no explicit files or SHA range, inject last known SHA for incremental
        if merged_job.get("changed_files") is None and not merged_job.get("before_sha"):
            last_sha = _get_last_sha(repo)
            if last_sha:
                merged_job["before_sha"] = last_sha
                print(f"[webhook] injecting before_sha={last_sha[:8]} from commit store")

        # Crash detector ON
        _mark_ingest_started(repo)

        try:
            commit_sha = ingest(**merged_job)
            if commit_sha:
                _mark_ingest_done(repo, commit_sha)
            else:
                print(f"[webhook] ingest returned None for repo={repo}; done stays false")
        except Exception as e:
            print(f"[webhook] ingest failed for repo={repo}: {e}. "
                  "done=false — will retry on next webhook or restart.")

        for job in all_jobs:
            _delete_job(repo, job)


def _enqueue(repo: str, job: dict):
    _persist_job(repo, job)
    state = _get_state(repo)
    state.queue.put(job)
    print(f"[webhook] enqueued job for repo={repo} queue_size={state.queue.qsize()}")


# ─────────────────────────────────────────────────────────────────────────────
# Job persistence
# ─────────────────────────────────────────────────────────────────────────────

def _job_file(repo: str) -> Path:
    return JOBS_DIR / f"{repo}.jsonl"


def _persist_job(repo: str, job: dict):
    with open(_job_file(repo), "a") as f:
        f.write(json.dumps({"job": job, "ts": time.time()}) + "\n")


def _delete_job(repo: str, job: dict):
    path = _job_file(repo)
    if not path.exists():
        return
    try:
        lines     = path.read_text().splitlines()
        remaining = [l for l in lines if l and json.loads(l).get("job") != job]
        if remaining:
            path.write_text("\n".join(remaining) + "\n")
        else:
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[webhook] _delete_job error for repo={repo}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Signature verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_github_signature(body: bytes, sig_header: str) -> bool:
    if not sig_header or not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)


# ─────────────────────────────────────────────────────────────────────────────
# GitHub webhook
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/github")
async def github_webhook(request: Request):
    token = request.query_params.get("token")
    body  = await request.body()
    sig   = request.headers.get("X-Hub-Signature-256", "")

    if WEBHOOK_SECRET and not verify_github_signature(body, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event   = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(body)

    if event == "push":
        repo_url = payload["repository"]["clone_url"]
        branch   = payload["ref"].replace("refs/heads/", "")
        repo     = _repo_name(repo_url)

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

        before_sha = payload.get("before")
        after_sha  = payload.get("after")

        print(f"[github] push → {repo_url} branch={branch} "
              f"changed={len(changed_files)} removed={len(removed_files)} "
              f"before={before_sha[:8] if before_sha else 'none'} "
              f"after={after_sha[:8] if after_sha else 'none'}")

        if removed_files:
            from qdrant_store import delete_chunks_for_files
            delete_chunks_for_files(repo, removed_files)

        if changed_files:
            known = _repo_is_known(repo)
            if not known:
                print(f"[github] repo={repo} unknown — forcing full re-ingest to clean Qdrant")

            _enqueue(repo, dict(
                repo_url=repo_url, branch=branch,
                changed_files=changed_files if known else None,
                before_sha=before_sha if known else None,
                after_sha=after_sha if known else None,
                token=token,
            ))

    return {"status": "queued"}


# ─────────────────────────────────────────────────────────────────────────────
# Bitbucket webhook
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/bitbucket")
async def bitbucket_webhook(request: Request):
    token   = request.query_params.get("token")
    body    = await request.body()
    payload = json.loads(body)
    event   = request.headers.get("X-Event-Key", "")

    if event == "repo:push":
        repo_url = payload["repository"].get("links", {}).get("html", {}).get("href")
        if not repo_url:
            repo_url = payload["repository"].get("full_name")
            if repo_url:
                repo_url = f"https://bitbucket.org/{repo_url}"

        repo = _repo_name(repo_url)

        for change in payload.get("push", {}).get("changes", []):
            branch     = change.get("new", {}).get("name", DEFAULT_BRANCH)
            before_sha = change.get("old", {}).get("target", {}).get("hash")
            after_sha  = change.get("new", {}).get("target", {}).get("hash")

            print(f"[bitbucket] push → {repo_url} branch={branch} "
                  f"before={before_sha[:8] if before_sha else 'none'} "
                  f"after={after_sha[:8] if after_sha else 'none'}")

            known = _repo_is_known(repo)
            if not known:
                print(f"[bitbucket] repo={repo} unknown — forcing full re-ingest to clean Qdrant")

            _enqueue(repo, dict(
                repo_url=repo_url, branch=branch,
                changed_files=None,  # Bitbucket never sends file list; always use SHA diff
                before_sha=before_sha if known else None,
                after_sha=after_sha if known else None,
                token=token,
            ))

    return {"status": "queued"}


# ─────────────────────────────────────────────────────────────────────────────
# Manual trigger
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/trigger")
async def manual_trigger(request: Request):
    body     = await request.json()
    repo_url = body.get("repo_url")
    branch   = body.get("branch", "main")
    full     = body.get("full", True)
    token    = body.get("token")

    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url required")

    repo = _repo_name(repo_url)
    _enqueue(repo, dict(
        repo_url=repo_url, branch=branch,
        changed_files=None if full else body.get("changed_files", []),
        token=token,
    ))
    return {"status": "queued", "repo_url": repo_url, "branch": branch}


# ─────────────────────────────────────────────────────────────────────────────
# Repo rename
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/rename")
async def rename_repo(request: Request):
    body    = await request.json()
    old_url = body.get("old_repo_url")
    new_url = body.get("new_repo_url")

    if not old_url or not new_url:
        raise HTTPException(status_code=400, detail="old_repo_url and new_repo_url required")

    collection = _repo_name(old_url)

    from qdrant_store import get_client
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if collection not in existing:
        raise HTTPException(status_code=404, detail=f"collection '{collection}' not found")

    updated, offset = 0, None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=500, offset=offset,
            with_payload=False, with_vectors=False,
        )
        if not results:
            break
        client.set_payload(
            collection_name=collection,
            payload={"repo_url": new_url},
            points=[r.id for r in results],
        )
        updated += len(results)
        if offset is None:
            break

    print(f"[webhook] repo rename: updated {updated} chunks in '{collection}'")

    old_path = LOCAL_REPOS_DIR / collection
    new_name = _repo_name(new_url)
    new_path = LOCAL_REPOS_DIR / new_name
    if old_path.exists() and old_path != new_path:
        shutil.move(str(old_path), str(new_path))
        print(f"[webhook] repo rename: '{old_path}' → '{new_path}'")

    with _commit_store_lock:
        try:
            store = json.loads(COMMIT_STORE.read_text()) if COMMIT_STORE.exists() else {}
        except Exception:
            store = {}
        if collection in store:
            store[new_name] = store.pop(collection)
            COMMIT_STORE.write_text(json.dumps(store, indent=2))

    return {"status": "done", "collection": collection,
            "chunks_updated": updated, "new_repo_url": new_url}


# ─────────────────────────────────────────────────────────────────────────────
# File rename
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/rename-file")
async def rename_file(request: Request):
    body          = await request.json()
    repo_url      = body.get("repo_url")
    old_file_path = body.get("old_file_path")
    new_file_path = body.get("new_file_path")

    if not repo_url or not old_file_path or not new_file_path:
        raise HTTPException(status_code=400,
                            detail="repo_url, old_file_path, new_file_path required")

    collection = _repo_name(repo_url)

    from qdrant_store import get_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if collection not in existing:
        raise HTTPException(status_code=404, detail=f"collection '{collection}' not found")

    updated, offset = 0, None
    file_filter = Filter(must=[
        FieldCondition(key="file_path", match=MatchValue(value=old_file_path))
    ])

    while True:
        results, offset = client.scroll(
            collection_name=collection,
            scroll_filter=file_filter,
            limit=500, offset=offset,
            with_payload=False, with_vectors=False,
        )
        if not results:
            break
        client.set_payload(
            collection_name=collection,
            payload={"file_path": new_file_path},
            points=[r.id for r in results],
        )
        updated += len(results)
        if offset is None:
            break

    print(f"[webhook] file rename: '{old_file_path}' → '{new_file_path}' "
          f"in '{collection}', {updated} chunks updated")

    return {"status": "done", "collection": collection,
            "old_file_path": old_file_path,
            "new_file_path": new_file_path,
            "chunks_updated": updated}


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    store = _load_commit_store()
    return {
        "status":       "ok",
        "active_repos": len(_repos),
        "queues":       {repo: s.queue.qsize() for repo, s in _repos.items()},
        "commit_store": {
            repo: {"sha": (e.get("sha") or "")[:8], "done": e.get("done")}
            for repo, e in store.items()
        },
    }