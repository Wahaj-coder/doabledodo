# search_eval_custom.py — search only, ingestion already done

import json
import requests

SEARCH_URL  = "http://localhost:8001/search"
OUTPUT_JSON = "semantic_eval_results.json"
OUTPUT_MD   = "semantic_eval_summary.md"

COLLECTION = "you-get"
TOP_K      = 15

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

total       = len(QUERY_SET)
recall_at_k = hits / total if total else 0
mrr         = mrr_total / total if total else 0

print(f"\n{'='*60}")
print(f"Recall@{TOP_K} : {recall_at_k:.3f}  ({hits}/{total})")
print(f"MRR           : {mrr:.3f}")
print(f"Errors        : {errors}")
print(f"{'='*60}")

output = {
    "config":  {"collection": COLLECTION, "top_k": TOP_K, "search_mode": "hybrid"},
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
print(f"Saved → {OUTPUT_JSON}")

misses = [r for r in results if not r["hit"]]
md = f"""# RAG Eval — you-get (Hybrid)

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
    top = r["retrieved"][0]["name"] if r["retrieved"] else "nothing"
    md += f"| {r['query'][:60]} | {r['target']} | {top} |\n"

with open(OUTPUT_MD, "w") as f:
    f.write(md)
print(f"Saved → {OUTPUT_MD}")