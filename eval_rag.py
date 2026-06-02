# search_eval_custom.py
# ----------------------------------------------------------
# 1. Trigger ingestion of you-get repo
# 2. Poll until done
# 3. Run hybrid search eval on hand-written queries
# 4. Save results
# ----------------------------------------------------------

import json
import time
import re
import requests

SEARCH_URL  = "http://localhost:8001/search"
TRIGGER_URL = "http://localhost:8000/trigger"
HEALTH_URL  = "http://localhost:8000/health"
OUTPUT_JSON = "semantic_eval_results.json"
OUTPUT_MD   = "semantic_eval_summary.md"

REPO_URL   = "https://github.com/soimort/you-get"
BRANCH     = "develop"
COLLECTION = "you-get"
TOP_K      = 15


def _repo_name(repo_url: str) -> str:
    clean = re.sub(r"https://[^@]+@", "https://", repo_url)
    name  = clean.rstrip("/").split("/")[-1].replace(".git", "")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


QUERY_SET = [
    {"target": "fc2video_download",                   "query": "where is fc2 video download wrapped"},
    {"target": "yixia_download",                      "query": "where is yixia video download function"},
    {"target": "vimeo_download_by_channel",           "query": "how to download vimeo channel videos"},
    {"target": "vimeo_download_by_channel_id",        "query": "download vimeo channel by channel id"},
    {"target": "dictify",                             "query": "where is html table converted to dictionary"},
    {"target": "veoh_download_by_id",                 "query": "how is veoh video downloaded by id"},
    {"target": "get_video_url_from_video_id",         "query": "how is ixigua video url built from video id"},
    {"target": "MGTV.get_vid_from_url",               "query": "where does mgtv extract video id from url"},
    {"target": "Iqiyi.download",                      "query": "where is iqiyi download method overridden"},
    {"target": "acfun_download_by_vid",               "query": "how is acfun video downloaded by vid"},
    {"target": "showroom_get_roomid_by_room_url_key", "query": "where is showroom room id extracted from url"},
]


# ─────────────────────────────────────────────────────────────
# Step 1: Trigger ingestion
# ─────────────────────────────────────────────────────────────

print(f"[eval] Triggering ingestion: {REPO_URL}")
try:
    resp = requests.post(TRIGGER_URL, json={
        "repo_url": REPO_URL,
        "branch":   BRANCH,
        "full":     True,
    }, timeout=30)
    print(f"[eval] Trigger response: {resp.json()}")
except Exception as e:
    print(f"[eval] Trigger failed: {e} — make sure ingestor service is running on :8000")
    exit(1)


# ─────────────────────────────────────────────────────────────
# Step 2: Poll until ingestion complete
# ─────────────────────────────────────────────────────────────

print(f"\n[eval] Sleeping 30s for worker to pick up job...")
time.sleep(30)

print(f"[eval] Polling until ingestion complete...")
while True:
    try:
        health  = requests.get(HEALTH_URL, timeout=5).json()
        queues  = health.get("queues", {})
        active  = health.get("active_repos", 0)
        pending = {r: q for r, q in queues.items() if q > 0}

        print(f"[eval] active_repos={active}  queues={queues}")

        if not pending and active == 0:
            print("[eval] ✅ Ingestion complete.")
            break

        print("[eval] Still running — waiting 20s...")
        time.sleep(20)

    except Exception as e:
        print(f"[eval] Health check failed: {e} — retrying in 10s...")
        time.sleep(10)


# ─────────────────────────────────────────────────────────────
# Step 3: Search eval — hybrid only
# ─────────────────────────────────────────────────────────────

print(f"\n[eval] Running hybrid search eval ({len(QUERY_SET)} queries, TOP_K={TOP_K})...")

results   = []
hits      = 0
mrr_total = 0.0
errors    = 0

for i, item in enumerate(QUERY_SET, 1):
    target = item["target"]
    query  = item["query"]

    print(f"\n[{i}/{len(QUERY_SET)}]")
    print(f"QUERY  : {query}")
    print(f"TARGET : {target}")

    try:
        resp = requests.post(
            SEARCH_URL,
            json={
                "query":          query,
                "collection":     COLLECTION,
                "top_k":          TOP_K,
                "use_rerank":     False,
                "expand_calls":   True,
                "called_by_hops": 2,
                "search_mode":    "hybrid",
            },
            timeout=60,
        )
        retrieved = resp.json().get("results", [])
    except Exception as e:
        print(f"ERROR: {e}")
        retrieved = []
        errors += 1

    # hit detection
    hit_rank     = None
    target_lower = target.lower()
    target_parts = target_lower.split(".")

    for rank, r in enumerate(retrieved, 1):
        name = (r.get("name") or "").lower()
        if any(part in name for part in target_parts):
            hit_rank = rank
            break

    hit = hit_rank is not None
    if hit:
        hits      += 1
        mrr_total += 1.0 / hit_rank

    print(f"HIT: {hit} | Rank: {hit_rank}")
    print("TOP 5:")
    for j, r in enumerate(retrieved[:5], 1):
        print(f"  {j}. {r.get('name','?')}  (score={r.get('score',0):.3f})")

    results.append({
        "target":   target,
        "query":    query,
        "hit":      hit,
        "hit_rank": hit_rank,
        "retrieved": [
            {
                "rank":      rank,
                "file_path": r.get("file_path"),
                "name":      r.get("name"),
                "score":     r.get("score"),
            }
            for rank, r in enumerate(retrieved, 1)
        ],
    })


# ─────────────────────────────────────────────────────────────
# Step 4: Metrics
# ─────────────────────────────────────────────────────────────

total       = len(QUERY_SET)
recall_at_k = hits / total if total else 0
mrr         = mrr_total / total if total else 0

print(f"\n{'='*60}")
print(f"Recall@{TOP_K} : {recall_at_k:.3f}  ({hits}/{total})")
print(f"MRR           : {mrr:.3f}")
print(f"Errors        : {errors}")
print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# Step 5: Save JSON
# ─────────────────────────────────────────────────────────────

output = {
    "config": {
        "repo":        REPO_URL,
        "collection":  COLLECTION,
        "top_k":       TOP_K,
        "search_mode": "hybrid",
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

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"[eval] Saved → {OUTPUT_JSON}")


# ─────────────────────────────────────────────────────────────
# Step 6: Save markdown
# ─────────────────────────────────────────────────────────────

misses = [r for r in results if not r["hit"]]

md = f"""# RAG Eval — you-get (Hybrid Search)

## Config
| Setting | Value |
|---|---|
| Repo | {REPO_URL} |
| Collection | {COLLECTION} |
| Top-K | {TOP_K} |
| Search Mode | hybrid |

## Metrics
| Metric | Score |
|---|---|
| Recall@{TOP_K} | {recall_at_k:.3f} |
| MRR | {mrr:.3f} |
| Hits | {hits}/{total} |
| Errors | {errors} |

## Failures
| Query | Target | Top Retrieved |
|---|---|---|
"""

for r in misses:
    top         = r["retrieved"][0]["name"] if r["retrieved"] else "nothing"
    query_short = r["query"][:60].replace("|", "/")
    md += f"| {query_short} | {r['target']} | {top} |\n"

with open(OUTPUT_MD, "w") as f:
    f.write(md)
print(f"[eval] Saved → {OUTPUT_MD}")