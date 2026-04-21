import os
import requests
from dotenv import load_dotenv
from qdrant_client.models import SearchRequest, NamedVector
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
MODEL = os.getenv("EMBEDDING_MODEL")

# ─────────────────────────────
# 1. CHECK OLLAMA
# ─────────────────────────────
print("\n[1] Checking Ollama...")

r = requests.get(OLLAMA_URL)
print("[✔] Ollama running")


# ─────────────────────────────
# 2. TEST EMBEDDING
# ─────────────────────────────
print("\n[2] Testing embedding...")

def embed(texts):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": MODEL, "prompt": texts[0]},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["embedding"]

vec = embed(["hello world"])
print("[✔] Vector dim:", len(vec))


# ─────────────────────────────
# 3. SIMPLE QDRANT TEST
# ─────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(path="./qdrant_data")

COLLECTION = "test"

print("\n[3] Creating collection...")

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE),
)

print("[✔] Collection ready")


# ─────────────────────────────
# 4. INSERT DATA
# ─────────────────────────────
print("\n[4] Inserting data...")

texts = [
    "Python is a programming language",
    "Qdrant is a vector database",
    "Ollama runs local LLMs"
]

points = []
for i, t in enumerate(texts):
    v = embed([t])
    points.append(
        PointStruct(id=i, vector=v, payload={"text": t})
    )

client.upsert(collection_name=COLLECTION, points=points)

print("[✔] Inserted")


# ─────────────────────────────
# 5. SEARCH TEST
# ─────────────────────────────
print("\n[5] Searching...")

query = "what is qdrant?"

q_vec = embed([query])

results = client.query_points(
    collection_name=COLLECTION,
    query=q_vec,
    limit=3
).points

for r in results:
    print(f"score={r.score:.3f} | {r.payload['text']}")

print("\n[✔] FULL PIPELINE WORKING")