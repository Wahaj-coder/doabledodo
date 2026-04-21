"""
main2.py - Full pipeline test
Uses: embedder.py, qdrant_store.py, ingestor.py, search_api logic
"""

import sys
import requests

sys.path.append(".")

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

OLLAMA_URL  = "http://localhost:11434"
REPO_URL    = "https://github.com/Wahaj-coder/abc___X"
BRANCH      = "main"
COLLECTION  = "test_repo"

# ==============================
# 1. CHECK OLLAMA
# ==============================
print("\n[1] Ollama check...")
requests.get(OLLAMA_URL)
print("[✔] Ollama running")

# ==============================
# 2. LOAD MODULES
# ==============================
print("\n[2] Loading modules...")
from embedder import embed_texts, embed_chunks
from ingestor import ingest
from qdrant_store import search, close_client
print("[✔] All modules loaded")

# ==============================
# 3. TEST EMBEDDER
# ==============================
print("\n[3] Testing embedder...")
test_chunks = [
    {
        "text":        "def add(a, b): return a + b",
        "doc_type":    "code",
        "name":        "add",
        "chunk_index": 0,
        "file_path":   "math.py"
    }
]
result_chunks = embed_chunks(test_chunks)
print("[✔] Vector dim:", len(result_chunks[0]["vector"]))

# ==============================
# 4. QDRANT INIT
# ==============================
print("\n[4] Qdrant setup...")
client = QdrantClient(path="./qdrant_data")
vec_dim = len(result_chunks[0]["vector"])

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vec_dim, distance=Distance.COSINE),
    )
print("[✔] Collection ready")

# ==============================
# 5. UPLOAD TEST CHUNK
# ==============================
print("\n[5] Uploading test chunk...")
client.upsert(COLLECTION, points=[
    PointStruct(
        id=0,
        vector=result_chunks[0]["vector"],
        payload={
            "text":      result_chunks[0]["text"],
            "name":      result_chunks[0]["name"],
            "file_path": result_chunks[0]["file_path"],
            "repo_url":  "local_test",
            "branch":    "main",
        }
    )
])
print("[✔] Chunk inserted")

# ==============================
# 6. SEARCH TEST (only local_test data)
# ==============================
print("\n[6] Search test (filtered to local_test)...")
q_vec = embed_texts(["how does addition work?"])[0]

results = client.query_points(
    collection_name=COLLECTION,
    query=q_vec,
    limit=3,
    query_filter=Filter(
        must=[FieldCondition(key="repo_url", match=MatchValue(value="local_test"))]
    )
).points

for r in results:
    print(f"{r.score:.3f} | {r.payload['text']}")

client.close()

# ==============================
# 7. INGEST GITHUB REPO
# ==============================
print("\n[7] GitHub ingestion...")
ingest(repo_url=REPO_URL, branch=BRANCH, changed_files=None, collection_name=COLLECTION)
close_client()  # release qdrant_store's lock
print("[✔] Repo ingested")

# ==============================
# 8. SEARCH REPO (using qdrant_store.search with filter)
# ==============================
print("\n[8] Search repo (filtered to repo_url)...")
q_vec = embed_texts(["what is done in cricket related method"])[0]

results = search(
    collection_name=COLLECTION,
    query_vector=q_vec,
    top_k=3,
    filter_repo=REPO_URL,
    filter_branch=BRANCH,
)

for r in results:
    print(f"\n[{r['score']:.3f}] {r['file_path']}")
    print(r['text'][:150])

close_client()

print("\n[✔] FULL PIPELINE COMPLETE")

# from qdrant_store import search, close_client
# from embedder import embed_texts

# REPO_URL   = "https://github.com/Wahaj-coder/abc___X"
# BRANCH     = "main"
# COLLECTION = "test_repo"

# q_vec = embed_texts(["cricket"])[0]

# results = search(
#     collection_name=COLLECTION,
#     query_vector=q_vec,
#     top_k=3,
#     filter_repo=REPO_URL,
#     filter_branch=BRANCH,
# )

# for r in results:
#     print(f"\n[{r['score']:.3f}] {r['file_path']}")
#     print(r['text'][:150])

# close_client()