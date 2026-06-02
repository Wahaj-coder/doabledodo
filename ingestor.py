"""
ingestor.py
-----------
Full ingest pipeline with hybrid (dense + BM25 sparse) indexing.

Changes vs previous version:
  - Detailed step-by-step logging so you can see exactly where it is at all times.
  - Per-repo BM25 vocab: load_bm25_encoder(collection) / save_bm25_encoder(collection)
    write to bm25_vocab_<collection>.json — no shared global file.
  - ingest() returns commit_sha so webhook_server can call _mark_ingest_done().
  - If before_sha is provided but after_sha is not, after_sha defaults to HEAD.
  - If before_sha == HEAD nothing changed → returns early with current SHA.
  - Passes existing_name_map from Qdrant into chunk_repo() for incremental runs.
"""

import os
import re
import subprocess
import argparse
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from chunker import chunk_repo
from embedder import embed_chunks, save_bm25_encoder, load_bm25_encoder
from qdrant_store import delete_chunks_for_files, upsert_chunks, get_name_map
from describer import describe_chunks, enrich_embed_text, patch_embedder

patch_embedder()

LOCAL_REPOS_DIR   = os.getenv("LOCAL_REPOS_DIR",   "./repos")
SKIP_DESCRIPTIONS = os.getenv("SKIP_DESCRIPTIONS", "true").lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_name(repo_url: str) -> str:
    clean = re.sub(r"https://[^@]+@", "https://", repo_url)
    name  = clean.rstrip("/").split("/")[-1].replace(".git", "")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _inject_token(repo_url: str, token: str) -> str:
    if "bitbucket.org" in repo_url:
        return repo_url.replace("https://", f"https://x-token-auth:{token}@")
    elif "github.com" in repo_url:
        username = repo_url.split("github.com/")[1].split("/")[0]
        return repo_url.replace("https://", f"https://{username}:{token}@")
    return repo_url


def clone_or_pull(repo_url: str, branch: str = "main", token: str = None) -> str:
    local_path = os.path.join(LOCAL_REPOS_DIR, _repo_name(repo_url))
    os.makedirs(LOCAL_REPOS_DIR, exist_ok=True)
    clone_url = _inject_token(repo_url, token) if token else repo_url
    if not os.path.exists(local_path):
        print(f"[ingestor] Cloning → {local_path}")
        subprocess.run(["git", "clone", "--branch", branch, clone_url, local_path], check=True)
    else:
        print(f"[ingestor] Pulling latest in {local_path}")
        if token:
            subprocess.run(["git", "-C", local_path, "remote", "set-url", "origin", clone_url], check=True)
        subprocess.run(["git", "-C", local_path, "pull"], check=True)
        if token:
            subprocess.run(["git", "-C", local_path, "remote", "set-url", "origin", repo_url], check=True)
    return local_path


def get_current_commit(local_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", local_path, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def get_changed_files(local_path: str, before_sha: str, after_sha: str) -> Optional[list]:
    try:
        result = subprocess.run(
            ["git", "-C", local_path, "diff", "--name-only", before_sha, after_sha],
            capture_output=True, text=True, check=True,
        )
        files = result.stdout.strip().splitlines()
        print(f"[ingestor] Incremental: {len(files)} files changed "
              f"({before_sha[:8]}..{after_sha[:8]})")
        return files if files else []
    except subprocess.CalledProcessError as e:
        print(f"[ingestor] Could not diff {before_sha[:8]}..{after_sha[:8]}: {e}. "
              "Falling back to full ingest.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main ingest
# ─────────────────────────────────────────────────────────────────────────────

def ingest(
    repo_url:        str,
    branch:          str  = "main",
    changed_files:   list = None,
    before_sha:      str  = None,
    after_sha:       str  = None,
    token:           str  = None,
    collection_name: str  = None,
) -> Optional[str]:
    """
    Run the full ingest pipeline. Returns the HEAD commit SHA on success.

    Incremental logic (in priority order):
      1. changed_files provided explicitly → use them directly.
      2. before_sha provided (after_sha optional) → git diff before_sha..HEAD.
      3. Neither → full re-ingest of every file.

    BM25 vocab is loaded/saved per collection (bm25_vocab_<collection>.json).
    """
    collection = collection_name or _repo_name(repo_url)

    # ── Step 1: clone / pull ──────────────────────────────────────────────────
    print(f"[ingestor] Step 1/7 — clone/pull repo={repo_url} branch={branch}")
    local_path = clone_or_pull(repo_url, branch, token=token)
    print(f"[ingestor] Step 1/7 done — local path: {local_path}")

    commit_sha = get_current_commit(local_path)
    print(f"[ingestor] Repo: {repo_url} | Collection: {collection} | HEAD: {commit_sha[:8]}")

    # ── Step 2: resolve what to index ────────────────────────────────────────
    print(f"[ingestor] Step 2/7 — resolving changed files "
          f"(changed_files={'explicit' if changed_files is not None else 'none'}, "
          f"before_sha={before_sha[:8] if before_sha else 'none'})")

    if changed_files is None and before_sha:
        effective_after = after_sha or commit_sha

        if before_sha == effective_after:
            print(f"[ingestor] Already at {commit_sha[:8]}, nothing to do.")
            return commit_sha

        changed_files = get_changed_files(local_path, before_sha, effective_after)
        if changed_files == []:
            print("[ingestor] Diff returned no files. Nothing to do.")
            return commit_sha

    print(f"[ingestor] Step 2/7 done — "
          f"{'full re-ingest' if changed_files is None else f'{len(changed_files)} files'}")

    # ── Step 3: delete stale chunks ───────────────────────────────────────────
    print(f"[ingestor] Step 3/7 — deleting stale chunks from Qdrant")
    if changed_files:
        print(f"[ingestor] Deleting old chunks for {len(changed_files)} files...")
        delete_chunks_for_files(collection, changed_files)
    else:
        print(f"[ingestor] Full re-ingest — Qdrant collection will be overwritten by upsert")
    print(f"[ingestor] Step 3/7 done")

    # ── Step 4: fetch name map for cross-file dep resolution ─────────────────
    print(f"[ingestor] Step 4/7 — fetching name map from Qdrant")
    existing_name_map = None
    if changed_files:
        existing_name_map = get_name_map(collection)
    print(f"[ingestor] Step 4/7 done")

    # ── Step 5: chunk ─────────────────────────────────────────────────────────
    print(f"[ingestor] Step 5/7 — chunking "
          f"{'changed files' if changed_files else 'entire repo'}...")
    chunks = chunk_repo(
        local_path,
        changed_files=changed_files,
        existing_name_map=existing_name_map,
    )
    print(f"[ingestor] Step 5/7 done — {len(chunks)} chunks produced")

    if not chunks:
        print("[ingestor] No chunks produced. Done.")
        return commit_sha

    for chunk in chunks:
        chunk["repo_url"]   = repo_url
        chunk["branch"]     = branch
        chunk["commit_sha"] = commit_sha

    # ── Descriptions (optional) ───────────────────────────────────────────────
    if SKIP_DESCRIPTIONS:
        print("[ingestor] SKIP_DESCRIPTIONS=true — skipping LLM descriptions")
    else:
        print("[ingestor] Generating LLM descriptions...")
        chunks = describe_chunks(chunks)
        chunks = enrich_embed_text(chunks)
        print("[ingestor] Descriptions done")

    # ── Step 6: embed ─────────────────────────────────────────────────────────
    print(f"[ingestor] Step 6/7 — embedding {len(chunks)} chunks")

    print(f"[ingestor] Step 6a — loading BM25 encoder for collection='{collection}'")
    load_bm25_encoder(collection)
    print(f"[ingestor] Step 6a done — BM25 encoder ready")

    print(f"[ingestor] Step 6b — calling embed_chunks (dense + sparse) ...")
    chunks_with_vectors = embed_chunks(chunks, collection=collection)
    print(f"[ingestor] Step 6b done — {len(chunks_with_vectors)} chunks embedded")

    print(f"[ingestor] Step 6c — saving BM25 vocab for collection='{collection}'")
    save_bm25_encoder(collection)
    print(f"[ingestor] Step 6c done")

    print(f"[ingestor] Step 6/7 done")

    # ── Step 7: upsert ────────────────────────────────────────────────────────
    print(f"[ingestor] Step 7/7 — upserting to Qdrant collection='{collection}'")
    upsert_chunks(collection, chunks_with_vectors)
    print(f"[ingestor] Step 7/7 done")

    print(f"[ingestor] ✅ Done. {len(chunks_with_vectors)} chunks indexed "
          f"into '{collection}' @ {commit_sha[:8]}")

    return commit_sha


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",       required=True)
    parser.add_argument("--branch",     default="main")
    parser.add_argument("--token",      default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--skip-descriptions", action="store_true")
    args = parser.parse_args()

    if args.no_llm:
        os.environ["USE_LLM_DESCRIPTIONS"] = "false"
    if args.skip_descriptions:
        os.environ["SKIP_DESCRIPTIONS"] = "true"
        SKIP_DESCRIPTIONS = True

    ingest(repo_url=args.repo, branch=args.branch,
           token=args.token, collection_name=args.collection)