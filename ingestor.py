"""
ingestor.py
-----------
Orchestrates the full ingest pipeline:
  1. Clone or pull repo via git
  2. Call chunker.py to get chunk dicts
  3. Call embedder.py to get vectors (auto-batched)
  4. Call qdrant_store.py to delete old + upsert new chunks

Called by:
  - cli.py          (first-time full ingest, CLI)
  - webhook_server.py (incremental, on push)

Usage (direct test):
    python ingestor.py --repo https://github.com/org/repo --branch main
"""

import os
import re
import subprocess
import argparse
from dotenv import load_dotenv

from chunker import chunk_repo
from embedder import embed_chunks
from qdrant_store import delete_chunks_for_files, upsert_chunks

load_dotenv()

LOCAL_REPOS_DIR = os.getenv("LOCAL_REPOS_DIR", "./repos")


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_name(repo_url: str) -> str:
    """Convert repo URL to a safe folder name."""
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def clone_or_pull(repo_url: str, branch: str = "main") -> str:
    """
    Clone repo if not present, otherwise pull latest.
    Returns local path to repo.
    """
    os.makedirs(LOCAL_REPOS_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_REPOS_DIR, _repo_name(repo_url))

    if not os.path.exists(local_path):
        print(f"[ingestor] Cloning {repo_url} → {local_path}")
        subprocess.run(
            ["git", "clone", "--branch", branch, repo_url, local_path],
            check=True
        )
    else:
        print(f"[ingestor] Pulling latest in {local_path}")
        subprocess.run(["git", "-C", local_path, "pull"], check=True)

    return local_path


def get_current_commit(local_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", local_path, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main ingest entry point
# ─────────────────────────────────────────────────────────────────────────────

def ingest(
    repo_url: str,
    branch: str = "main",
    changed_files: list[str] = None,   # None = full ingest, list = incremental
    collection_name: str = None,
):
    """
    Full pipeline: git → chunk → embed → qdrant upsert.

    Args:
        repo_url:        Git remote URL
        branch:          Branch to clone/pull
        changed_files:   If None, index entire repo.
                         If list, only re-index those files
                         (old chunks for those files are deleted first).
        collection_name: Qdrant collection name. Defaults to repo name.
    """
    collection = collection_name or _repo_name(repo_url)

    # ── 1. Git clone / pull ──────────────────────────────────────────────────
    local_path = clone_or_pull(repo_url, branch)
    commit_sha = get_current_commit(local_path)
    print(f"[ingestor] Commit: {commit_sha[:8]}")

    # ── 2. Delete old chunks for changed files ───────────────────────────────
    if changed_files:
        print(f"[ingestor] Deleting old chunks for {len(changed_files)} changed files...")
        delete_chunks_for_files(collection, changed_files)

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    print(f"[ingestor] Chunking {'changed files' if changed_files else 'entire repo'}...")
    chunks = chunk_repo(local_path, changed_files=changed_files)

    if not chunks:
        print("[ingestor] No chunks produced. Done.")
        return

    print(f"[ingestor] {len(chunks)} chunks produced.")

    # ── 4. Add metadata to each chunk ────────────────────────────────────────
    for chunk in chunks:
        chunk["repo_url"]   = repo_url
        chunk["branch"]     = branch
        chunk["commit_sha"] = commit_sha

    # ── 5. Embed (auto-batched inside embedder.py) ───────────────────────────
    print("[ingestor] Embedding chunks...")
    chunks_with_vectors = embed_chunks(chunks)

    # ── 6. Upsert to Qdrant ──────────────────────────────────────────────────
    print("[ingestor] Upserting to Qdrant...")
    upsert_chunks(collection, chunks_with_vectors)

    print(f"[ingestor] ✅ Done. {len(chunks_with_vectors)} chunks indexed into '{collection}'.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (for testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a repo into Qdrant")
    parser.add_argument("--repo",       required=True,  help="Git repo URL")
    parser.add_argument("--branch",     default="main", help="Branch name")
    parser.add_argument("--collection", default=None,   help="Qdrant collection name")
    args = parser.parse_args()

    ingest(
        repo_url=args.repo,
        branch=args.branch,
        collection_name=args.collection,
    )
