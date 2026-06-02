# xlcost_ingest.py
# Steps:
# 1. Check if already ingested in Qdrant -- skip everything if so
# 2. Download XLCoST from HuggingFace via Parquet
# 3. Sample N chunks per language
# 4. Write each chunk as a real code file into a local git repo
# 5. Save corpus index to xlcost_corpus.json  (eval script reads this)
# 6. Trigger ingestion via POST /trigger  (same as you-get script)
# 7. Poll health until done

import json
import random
import subprocess
import time
import urllib.parse
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

TRIGGER_URL  = "http://localhost:8000/trigger"
HEALTH_URL   = "http://localhost:8000/health"
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333

CORPUS_FILE     = "xlcost_corpus.json"
COLLECTION      = "xlcost"
SAMPLE_PER_LANG = 200
RANDOM_SEED     = 42

REPO_DIR = Path("C:/Users/Dell/Downloads/files/xlcost_repo").resolve()

LANG_SUBSETS = {
    "Python":     ("Python-program-level",     ".py"),
    "Java":       ("Java-program-level",       ".java"),
    "Cpp":        ("C++-program-level",        ".cpp"),
    "Javascript": ("Javascript-program-level", ".js"),
    "Csharp":     ("Csharp-program-level",     ".cs"),
    "PHP":        ("PHP-program-level",        ".php"),
    "C":          ("C-program-level",          ".c"),
}
 
HF_PARQUET_BASE = (
    "https://huggingface.co/datasets/codeparrot/xlcost-text-to-code"
    "/resolve/refs%2Fconvert%2Fparquet"
)

# ---------------------------------------------------------------

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------
# Step 1: Skip if already ingested
# ---------------------------------------------------------------

def collection_exists(name):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        cols = [c.name for c in client.get_collections().collections]
        return name in cols
    except Exception as e:
        print(f"[ingest] Could not check Qdrant ({e}) -- will proceed with trigger")
        return False


if collection_exists(COLLECTION):
    print(f"[ingest] Collection '{COLLECTION}' already exists in Qdrant.")
    print(f"[ingest] Skipping -- run xlcost_eval.py directly.")
    print(f"")
    print(f"[ingest] To re-ingest from scratch, delete the collection first:")
    print(f"  python -c \"from qdrant_client import QdrantClient; QdrantClient('localhost', 6333).delete_collection('{COLLECTION}')\"")
    exit(0)


# ---------------------------------------------------------------
# Step 2: Download + sample from HuggingFace
# ---------------------------------------------------------------

def load_parquet(subset_name, split="test"):
    encoded = urllib.parse.quote(subset_name, safe="")
    for shard in ["0000", "0001", "0002"]:
        url = f"{HF_PARQUET_BASE}/{encoded}/{split}/{shard}.parquet"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                return pd.read_parquet(BytesIO(resp.content))
        except Exception:
            pass
    if split == "test":
        return load_parquet(subset_name, split="train")
    return pd.DataFrame()


print(f"[ingest] Downloading XLCoST ({SAMPLE_PER_LANG} samples/lang)\n")

corpus = []

for lang, (subset, ext) in LANG_SUBSETS.items():
    print(f"[ingest] Fetching {subset} ...")
    df = load_parquet(subset, split="test")

    if df.empty:
        print(f"  -- Skipping {lang} -- no data")
        continue

    text_col = next((c for c in df.columns if c in ("text", "comment", "nl")), None)
    code_col = next((c for c in df.columns if c in ("code", "snippet")), None)

    if not text_col or not code_col:
        print(f"  -- Skipping {lang} -- unexpected columns {list(df.columns)}")
        continue

    pool    = df[[text_col, code_col]].dropna().to_dict("records")
    n       = min(SAMPLE_PER_LANG, len(pool))
    sampled = random.sample(pool, n)

    for idx, item in enumerate(sampled):
        chunk_id = f"{lang}_{idx:04d}"
        rel_path = f"{lang}/{chunk_id}{ext}"
        corpus.append({
            "chunk_id":  chunk_id,
            "language":  lang,
            "query":     str(item[text_col]).strip(),
            "code":      str(item[code_col]).strip(),
            "file_path": rel_path,
        })

    print(f"  OK {lang}: {n} chunks  (pool={len(pool)})")

print(f"\n[ingest] Total: {len(corpus)} chunks across {len(LANG_SUBSETS)} languages")


# ---------------------------------------------------------------
# Step 3: Write code files into a local git repo
# ---------------------------------------------------------------

print(f"\n[ingest] Writing files to {REPO_DIR} ...")

REPO_DIR.mkdir(parents=True, exist_ok=True)

if not (REPO_DIR / ".git").exists():
    subprocess.run(["git", "init", str(REPO_DIR)], check=True)
    subprocess.run(["git", "config", "user.email", "xlcost@eval.local"], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "config", "user.name",  "xlcost-eval"],       cwd=str(REPO_DIR), check=True)

comment_chars = {
    ".py": "#", ".java": "//", ".cpp": "//",
    ".js": "//", ".cs":  "//", ".php": "//", ".c": "//",
}

for chunk in corpus:
    dest = REPO_DIR / chunk["file_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    char    = comment_chars.get(Path(chunk["file_path"]).suffix, "#")
    content = f"{char} {chunk['query']}\n\n{chunk['code']}\n"
    dest.write_text(content, encoding="utf-8")

subprocess.run(["git", "add", "."], cwd=str(REPO_DIR), check=True)
subprocess.run(
    ["git", "commit", "-m", f"XLCoST benchmark -- {len(corpus)} chunks"],
    cwd=str(REPO_DIR), check=False,
)
subprocess.run(["git", "branch", "-M", "main"], cwd=str(REPO_DIR), check=False)

print(f"[ingest] Repo ready at {REPO_DIR}")


# ---------------------------------------------------------------
# Step 4: Save corpus JSON for eval script
# ---------------------------------------------------------------

with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

print(f"[ingest] Corpus index saved to {CORPUS_FILE}")


# ---------------------------------------------------------------
# Step 5: Trigger ingestion
# ---------------------------------------------------------------

repo_url = REPO_DIR.as_uri()
print(f"\n[ingest] Triggering ingestion: {repo_url}")

try:
    resp = requests.post(TRIGGER_URL, json={
        "repo_url":   repo_url,
        "branch":     "main",
        "full":       True,
        "collection": COLLECTION,
    }, timeout=30)
    print(f"[ingest] Trigger response: {resp.json()}")
except Exception as e:
    print(f"[ingest] Trigger failed: {e}")
    exit(1)


# ---------------------------------------------------------------
# Step 6: Poll health until done
# ---------------------------------------------------------------

print(f"\n[ingest] Sleeping 30s for worker to pick up job...")
time.sleep(30)

print(f"[ingest] Polling until ingestion complete...")
while True:
    try:
        health  = requests.get(HEALTH_URL, timeout=5).json()
        queues  = health.get("queues", {})
        active  = health.get("active_repos", 0)
        pending = {r: q for r, q in queues.items() if q > 0}

        print(f"[ingest] active_repos={active}  queues={queues}")

        if not pending and active == 0:
            print("[ingest] Ingestion complete -- run xlcost_eval.py next.")
            break

        print("[ingest] Still running -- waiting 20s...")
        time.sleep(20)

    except Exception as e:
        print(f"[ingest] Health check error: {e} -- retrying in 10s...")
        time.sleep(10)