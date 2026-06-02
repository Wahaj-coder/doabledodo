# xlcost_eval.py
# Reads xlcost_corpus.json produced by xlcost_ingest.py
# Runs search eval per language + combined
# Saves results to JSON + Markdown

import json
import requests
from collections import defaultdict

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

SEARCH_URL  = "http://localhost:8001/search"
CORPUS_FILE = "xlcost_corpus.json"
OUTPUT_JSON = "xlcost_eval_results.json"
OUTPUT_MD   = "xlcost_eval_summary.md"

COLLECTION       = "xlcost_repo"   # must match what Qdrant actually has
TOP_K            = 15
SEARCH_MODE      = "hybrid"        # "hybrid" | "semantic" | "keyword"
QUERIES_PER_LANG = 100              # set to None to run all

# ---------------------------------------------------------------
# Step 1: Load corpus
# ---------------------------------------------------------------

with open(CORPUS_FILE, encoding="utf-8") as f:
    corpus = json.load(f)

print(f"[eval] Loaded {len(corpus)} chunks from {CORPUS_FILE}")

by_lang = defaultdict(list)
for chunk in corpus:
    by_lang[chunk["language"]].append(chunk)

query_set = []
for lang, chunks in by_lang.items():
    pool = chunks if QUERIES_PER_LANG is None else chunks[:QUERIES_PER_LANG]
    query_set.extend(pool)

print(f"[eval] Running eval on {len(query_set)} queries across {len(by_lang)} languages")
print(f"[eval] Languages: {list(by_lang.keys())}\n")


# ---------------------------------------------------------------
# Step 2: Run search
# ---------------------------------------------------------------

results = []
errors  = 0

for i, chunk in enumerate(query_set, 1):
    lang        = chunk["language"]
    query       = chunk["query"]
    target_id   = chunk["chunk_id"]    # e.g. "Java_0042"
    target_file = chunk["file_path"]   # e.g. "Java/Java_0042.java"

    print(f"[{i}/{len(query_set)}] [{lang}] {query[:70]}")

    resp      = None
    retrieved = []

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
                "search_mode":    SEARCH_MODE,
            },
            timeout=60,
        )
        retrieved = resp.json().get("results", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        if resp is not None:
            print(f"  RAW: {resp.text[:300]}")
        errors += 1

    # debug first query -- shows what fields the search actually returns
    if i == 1 and retrieved:
        print(f"  [DEBUG] result keys  : {list(retrieved[0].keys())}")
        print(f"  [DEBUG] first result : {retrieved[0]}")
        print(f"  [DEBUG] target_file  : {target_file}")

    # hit detection -- match on file_path
    # ingestor stores file_path as "Java/Java_0042.java"
    # our target_file is also "Java/Java_0042.java"
    hit_rank = None
    for rank, r in enumerate(retrieved, 1):
        result_file = (r.get("file_path") or "").replace("\\", "/").lower()
        target_norm = target_file.replace("\\", "/").lower()
        if target_norm in result_file or result_file.endswith(target_norm):
            hit_rank = rank
            break

    hit = hit_rank is not None
    print(f"  HIT: {hit} | Rank: {hit_rank}")

    results.append({
        "language":    lang,
        "chunk_id":    target_id,
        "target_file": target_file,
        "query":       query,
        "hit":         hit,
        "hit_rank":    hit_rank,
        "retrieved": [
            {
                "rank":      rank,
                "name":      r.get("name"),
                "file_path": r.get("file_path"),
                "score":     r.get("score"),
            }
            for rank, r in enumerate(retrieved, 1)
        ],
    })


# ---------------------------------------------------------------
# Step 3: Metrics -- overall + per language
# ---------------------------------------------------------------

def compute_metrics(subset):
    total     = len(subset)
    hits      = sum(1 for r in subset if r["hit"])
    mrr_total = sum(1.0 / r["hit_rank"] for r in subset if r["hit"])
    return {
        f"recall_at_{TOP_K}": round(hits / total, 4) if total else 0,
        "mrr":                round(mrr_total / total, 4) if total else 0,
        "hits":               hits,
        "total":              total,
    }

overall  = compute_metrics(results)
per_lang = {}
for lang in by_lang:
    subset = [r for r in results if r["language"] == lang]
    if subset:
        per_lang[lang] = compute_metrics(subset)

print(f"\n{'='*60}")
print(f"OVERALL")
print(f"  Recall@{TOP_K} : {overall[f'recall_at_{TOP_K}']:.3f}  ({overall['hits']}/{overall['total']})")
print(f"  MRR           : {overall['mrr']:.3f}")
print(f"  Errors        : {errors}")
print(f"\nPER LANGUAGE:")
for lang, m in per_lang.items():
    print(f"  {lang:<12} Recall@{TOP_K}={m[f'recall_at_{TOP_K}']:.3f}  MRR={m['mrr']:.3f}  ({m['hits']}/{m['total']})")
print(f"{'='*60}")


# ---------------------------------------------------------------
# Step 4: Save JSON
# ---------------------------------------------------------------

output = {
    "config": {
        "collection":  COLLECTION,
        "top_k":       TOP_K,
        "search_mode": SEARCH_MODE,
        "corpus_file": CORPUS_FILE,
    },
    "metrics": {
        "overall":  overall,
        "per_lang": per_lang,
        "errors":   errors,
    },
    "results": results,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n[eval] Saved -> {OUTPUT_JSON}")


# ---------------------------------------------------------------
# Step 5: Save Markdown
# ---------------------------------------------------------------

misses = [r for r in results if not r["hit"]]

lang_table = "\n".join(
    f"| {lang} | {m[f'recall_at_{TOP_K}']:.3f} | {m['mrr']:.3f} | {m['hits']}/{m['total']} |"
    for lang, m in per_lang.items()
)

miss_table = "\n".join(
    f"| {r['language']} | {r['query'][:60]} | {r['target_file']} | {r['retrieved'][0]['file_path'] if r['retrieved'] else 'nothing'} |"
    for r in misses[:30]
)

md = f"""# XLCoST Eval -- {SEARCH_MODE.title()} Search

## Config
| Setting | Value |
|---|---|
| Collection | {COLLECTION} |
| Top-K | {TOP_K} |
| Search Mode | {SEARCH_MODE} |
| Queries per language | {QUERIES_PER_LANG or 'all'} |
| Total queries | {overall['total']} |

## Overall Metrics
| Metric | Score |
|---|---|
| Recall@{TOP_K} | {overall[f'recall_at_{TOP_K}']:.3f} |
| MRR | {overall['mrr']:.3f} |
| Hits | {overall['hits']}/{overall['total']} |
| Errors | {errors} |

## Per-Language Breakdown
| Language | Recall@{TOP_K} | MRR | Hits |
|---|---|---|---|
{lang_table}

## Failures (first 30)
| Lang | Query | Target File | Top Retrieved |
|---|---|---|---|
{miss_table}
"""

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"[eval] Saved -> {OUTPUT_MD}")