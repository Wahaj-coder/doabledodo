"""
eval_rag.py
-----------
Full end-to-end RAG evaluation script.

Flow:
  1. Load CodeSearchNet dataset
  2. Trigger ingestion for repos
  3. Wait until ingestion stabilizes
  4. Run retrieval evaluation
  5. Save JSON + markdown summary

Install:
  pip install datasets requests
"""

import json
import time
import re
import requests
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SEARCH_URL  = "http://localhost:8001/search"
TRIGGER_URL = "http://localhost:8000/trigger"
HEALTH_URL  = "http://localhost:8000/health"

LANGUAGE    = "python"
SAMPLE_SIZE = 50
MAX_REPOS   = 3
TOP_K       = 5
USE_RERANK  = False

OUTPUT_JSON = "eval_results2less.json"
OUTPUT_MD   = "eval_summary2less.md"

INITIAL_WAIT_SECONDS = 30
POLL_INTERVAL        = 10
MAX_STABLE_CHECKS    = 3


# ─────────────────────────────────────────────────────────────
# Repo helper
# ─────────────────────────────────────────────────────────────

def _repo_name(repo_url: str) -> str:
    clean = re.sub(r"https://[^@]+@", "https://", repo_url)
    name  = clean.rstrip("/").split("/")[-1].replace(".git", "")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ─────────────────────────────────────────────────────────────
# Step 1: Load dataset
# ─────────────────────────────────────────────────────────────

print(f"[eval] Loading CodeSearchNet '{LANGUAGE}' test split...")

ds = load_dataset(
    "code_search_net",
    LANGUAGE,
    split="test"
)

print(f"[eval] Dataset fields:")
print(list(ds[0].keys()))

all_fields = list(ds[0].keys())

FILE_FIELD = next(
    (
        f for f in
        ["func_path", "url", "func_code_url", "repo"]
        if f in all_fields
    ),
    None
)

NAME_FIELD = (
    "func_name"
    if "func_name" in all_fields
    else all_fields[0]
)

print(f"[eval] FILE_FIELD={FILE_FIELD}")
print(f"[eval] NAME_FIELD={NAME_FIELD}")


# ─────────────────────────────────────────────────────────────
# Step 2: Build samples
# ─────────────────────────────────────────────────────────────

samples = [
    s for s in list(
        ds.select(range(min(SAMPLE_SIZE * 5, len(ds))))
    )
    if s.get("func_documentation_string", "").strip()
][:SAMPLE_SIZE]

print(f"[eval] Initial samples: {len(samples)}")

repo_names = list(
    dict.fromkeys(
        s["repository_name"]
        for s in samples
    )
)[:MAX_REPOS]

repo_urls = [
    f"https://github.com/{r}"
    for r in repo_names
]

samples = [
    s for s in samples
    if s["repository_name"] in repo_names
]

print(f"[eval] Filtered samples: {len(samples)}")
print(f"[eval] Repos:")
for r in repo_urls:
    print(f"  - {r}")


# ─────────────────────────────────────────────────────────────
# Step 3: Trigger ingestion
# ─────────────────────────────────────────────────────────────

print(f"\n[eval] Triggering ingestion...")

for repo_url in repo_urls:
    try:
        resp = requests.post(
            TRIGGER_URL,
            json={
                "repo_url": repo_url,
                "branch":   "master",
                "full":     True,
            },
            timeout=20
        )

        try:
            data = resp.json()
        except:
            data = resp.text

        print(f"[eval] {repo_url}")
        print(f"        -> {data}")

    except Exception as e:
        print(f"[eval] Trigger failed for {repo_url}: {e}")


# ─────────────────────────────────────────────────────────────
# Step 4: Wait for ingestion completion
# ─────────────────────────────────────────────────────────────

print(f"\n[eval] Waiting {INITIAL_WAIT_SECONDS}s for workers...")
time.sleep(INITIAL_WAIT_SECONDS)

print(f"[eval] Polling health endpoint...")

stable_count = 0

while True:
    try:
        health = requests.get(
            HEALTH_URL,
            timeout=5
        ).json()

        queues = health.get("queues", {})
        active = health.get("active_repos", 0)

        pending = {
            repo: q
            for repo, q in queues.items()
            if q > 0
        }

        print(
            f"[eval] active_repos={active} "
            f"queues={queues}"
        )

        if active == 0 and not pending:
            stable_count += 1

            print(
                f"[eval] Stable check "
                f"{stable_count}/{MAX_STABLE_CHECKS}"
            )

            if stable_count >= MAX_STABLE_CHECKS:
                print("[eval] ✅ Ingestion complete")
                break

        else:
            stable_count = 0

        time.sleep(POLL_INTERVAL)

    except Exception as e:
        print(f"[eval] Health check failed: {e}")
        time.sleep(POLL_INTERVAL)


# ─────────────────────────────────────────────────────────────
# Step 5: Evaluate retrieval
# ─────────────────────────────────────────────────────────────

print(f"\n[eval] Starting evaluation...")
print(f"[eval] Queries: {len(samples)}")
print(f"[eval] TOP_K={TOP_K}")

hits      = 0
mrr_total = 0.0
errors    = 0
results   = []

for i, sample in enumerate(samples):

    query = sample["func_documentation_string"].strip()

    expected_file = (
        sample.get(FILE_FIELD, "")
        if FILE_FIELD else ""
    )

    expected_name = sample.get(NAME_FIELD, "")

    repo_url = (
        f"https://github.com/"
        f"{sample['repository_name']}"
    )

    collection = _repo_name(repo_url)

    try:
        resp = requests.post(
            SEARCH_URL,
            json={
                "query":      query,
                "collection": collection,
                "top_k":      TOP_K,
                "use_rerank": USE_RERANK,
            },
            timeout=30
        )

        retrieved = resp.json().get("results", [])

    except Exception as e:
        print(f"[eval] Search failed on sample {i}: {e}")

        retrieved = []
        errors += 1

    hit = False
    hit_rank = None

    for rank, r in enumerate(retrieved, 1):

        file_path = r.get("file_path") or ""
        name      = r.get("name") or ""

        file_match = (
            expected_file
            and expected_file in file_path
        )

        name_match = (
            expected_name
            and expected_name in name
        )

        if file_match or name_match:
            hit = True
            hit_rank = rank
            break

    if hit:
        hits += 1
        mrr_total += 1.0 / hit_rank

    results.append({
        "query":         query,
        "expected_file": expected_file,
        "expected_name": expected_name,
        "collection":    collection,
        "hit":           hit,
        "hit_rank":      hit_rank,
        "retrieved": [
            {
                "rank":      rank,
                "file_path": r.get("file_path"),
                "name":      r.get("name"),
                "score":     r.get("score"),
            }
            for rank, r in enumerate(retrieved, 1)
        ]
    })

    if (i + 1) % 10 == 0:
        print(
            f"[eval] {i+1}/{len(samples)} "
            f"hits={hits}"
        )


# ─────────────────────────────────────────────────────────────
# Step 6: Metrics
# ─────────────────────────────────────────────────────────────

total = len(results)

recall_at_k = (
    hits / total
    if total else 0
)

mrr = (
    mrr_total / total
    if total else 0
)

print("\n" + "=" * 60)
print(f"Recall@{TOP_K}: {recall_at_k:.4f}")
print(f"MRR          : {mrr:.4f}")
print(f"Hits         : {hits}/{total}")
print(f"Errors       : {errors}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Step 7: Save JSON
# ─────────────────────────────────────────────────────────────

output = {
    "config": {
        "language":    LANGUAGE,
        "sample_size": SAMPLE_SIZE,
        "max_repos":   MAX_REPOS,
        "top_k":       TOP_K,
        "use_rerank":  USE_RERANK,
        "repos":       repo_urls,
        "file_field":  FILE_FIELD,
        "name_field":  NAME_FIELD,
    },
    "metrics": {
        f"recall_at_{TOP_K}": round(recall_at_k, 4),
        "mrr":                round(mrr, 4),
        "hits":               hits,
        "total":              total,
        "errors":             errors,
    },
    "results": results,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"[eval] Saved JSON -> {OUTPUT_JSON}")


# ─────────────────────────────────────────────────────────────
# Step 8: Save markdown summary
# ─────────────────────────────────────────────────────────────

misses = [
    r for r in results
    if not r["hit"]
]

md = f"""# RAG Evaluation Results

## Config

| Setting | Value |
|---|---|
| Language | {LANGUAGE} |
| Queries | {total} |
| Repos | {len(repo_urls)} |
| Top-K | {TOP_K} |
| Reranker | {USE_RERANK} |

## Metrics

| Metric | Score |
|---|---|
| Recall@{TOP_K} | {recall_at_k:.4f} |
| MRR | {mrr:.4f} |
| Hits | {hits}/{total} |
| Errors | {errors} |

## Repos Tested

{chr(10).join(f'- {r}' for r in repo_urls)}

## Sample Failures

| Query | Expected | Top Retrieved |
|---|---|---|
"""

for r in misses[:10]:

    top = (
        r["retrieved"][0]["file_path"]
        if r["retrieved"]
        else "nothing"
    )

    query_short = (
        r["query"][:60]
        .replace("\n", " ")
        .replace("|", "/")
    )

    md += (
        f"| {query_short}... "
        f"| {r['expected_name']} "
        f"| {top} |\n"
    )

with open(OUTPUT_MD, "w") as f:
    f.write(md)

print(f"[eval] Saved markdown -> {OUTPUT_MD}")

print("\n[eval] ✅ Finished.")