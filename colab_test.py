# ============================================================
# RAG Pipeline - Google Colab Test Notebook
# ============================================================
# Run each cell in order.
# This tests: Ollama (via ngrok or local) → Qdrant → embedder → qdrant_store
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────────────────────
# Run this first, then restart runtime if asked

# !pip install qdrant-client requests python-dotenv fastapi uvicorn


# ── CELL 2: Mount Google Drive (for model storage) ──────────────────────────
# Qwen3-Embedding model files will be stored in Drive so you don't re-download

from google.colab import drive
drive.mount('/content/drive')

import os
MODEL_DIR = "/content/drive/MyDrive/ollama_models"
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model dir: {MODEL_DIR}")


# ── CELL 3: Install and configure Ollama in Colab ───────────────────────────
# Ollama installs to /usr/local and stores models in ~/.ollama by default.
# We symlink model storage to Drive so models persist across sessions.

import subprocess, os

# Install Ollama
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

# Symlink model storage to Drive (so models persist)
OLLAMA_MODELS_PATH = os.path.expanduser("~/.ollama/models")
os.makedirs(os.path.dirname(OLLAMA_MODELS_PATH), exist_ok=True)

if not os.path.exists(OLLAMA_MODELS_PATH):
    # First time: create in Drive, symlink here
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.symlink(MODEL_DIR, OLLAMA_MODELS_PATH)
    print(f"[setup] Symlinked {OLLAMA_MODELS_PATH} → {MODEL_DIR}")
else:
    print(f"[setup] {OLLAMA_MODELS_PATH} already exists (models cached from Drive)")

print("[setup] Ollama installed ✓")


# ── CELL 4: Start Ollama server + pull embedding model ──────────────────────
import subprocess, time

# Start Ollama in background
proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(3)
print("[ollama] Server started (PID:", proc.pid, ")")

# Pull model — skipped if already in Drive cache
# Use nomic-embed-text (small, fast, 768-dim) for testing
# Switch to qwen3-embedding for production
MODEL = "nomic-embed-text"   # or "qwen3:embedding" if available

result = subprocess.run(["ollama", "pull", MODEL], capture_output=True, text=True)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode == 0:
    print(f"[ollama] Model '{MODEL}' ready ✓")
else:
    print(f"[ollama] Pull error: {result.stderr[-300:]}")


# ── CELL 5: Test embedding directly ─────────────────────────────────────────
import requests

def test_embed(texts):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": MODEL, "input": texts},
        timeout=60
    )
    data = r.json()
    vecs = data["embeddings"]
    print(f"[test] Got {len(vecs)} vectors, dim={len(vecs[0])}")
    return vecs

vecs = test_embed([
    "def hello(): print('hello world')",
    "SELECT * FROM users WHERE active=true",
])
print("[test] ✅ Embedding works!")


# ── CELL 6: Upload your pipeline files ──────────────────────────────────────
# Upload chunker.py, embedder.py, qdrant_store.py to Colab
# Either use the Files panel on the left, or run this cell to upload

from google.colab import files
print("Upload your .py files: chunker.py, embedder.py, qdrant_store.py")
# uploaded = files.upload()   # uncomment to use file picker


# ── CELL 7: Set environment variables ───────────────────────────────────────
import os

os.environ["OLLAMA_BASE_URL"]  = "http://localhost:11434"
os.environ["EMBEDDING_MODEL"]  = MODEL
os.environ["QDRANT_MODE"]      = "local"
os.environ["QDRANT_LOCAL_DIR"] = "/content/qdrant_data"
os.environ["VECTOR_DIM"]       = "768"   # nomic=768, qwen3-embedding=1024

print("[env] Environment set ✓")


# ── CELL 8: Test embedder.py ─────────────────────────────────────────────────
# Make sure embedder.py is uploaded/in the same directory

import sys
sys.path.insert(0, "/content")   # or wherever you uploaded the files

from embedder import embed_texts, embed_chunks

test_chunks = [
    {"text": "def add(a, b): return a + b", "doc_type": "code", "name": "add", "chunk_index": 0, "file_path": "math.py"},
    {"text": "# README\nThis project does X", "doc_type": "markdown", "name": "intro", "chunk_index": 0, "file_path": "README.md"},
]

result_chunks = embed_chunks(test_chunks)
print(f"[test] Chunks with vectors: {len(result_chunks)}")
print(f"[test] Vector dim: {len(result_chunks[0]['vector'])}")
print("[test] ✅ embedder.py works!")


# ── CELL 9: Test qdrant_store.py ─────────────────────────────────────────────
from qdrant_store import create_collection_if_missing, upsert_chunks, search, delete_chunks_for_files

COLLECTION = "test_repo"

# Create collection
create_collection_if_missing(COLLECTION, vector_dim=768)
print("[test] Collection created ✓")

# Upsert test chunks (already have vectors from Cell 8)
upsert_chunks(COLLECTION, result_chunks.copy())
print("[test] Upsert done ✓")

# Search
from embedder import embed_texts
q_vec = embed_texts(["how does addition work?"])[0]
results = search(COLLECTION, q_vec, top_k=2)

print(f"\n[test] Search results ({len(results)}):")
for r in results:
    print(f"  score={r['score']:.3f}  file={r['file_path']}  name={r['name']}")
    print(f"  text: {r['text'][:80]}")

print("\n[test] ✅ qdrant_store.py works!")


# ── CELL 10: Full end-to-end test with a real repo ───────────────────────────
# Test the full pipeline: clone → chunk → embed → store → search

# Upload ingestor.py first, then run:
import sys
sys.path.insert(0, "/content")

from ingestor import ingest

TEST_REPO   = "https://github.com/pallets/click"   # small public repo
COLLECTION  = "click_repo"

ingest(
    repo_url=TEST_REPO,
    branch="main",
    changed_files=None,       # full ingest
    collection_name=COLLECTION,
)
print(f"\n[test] ✅ Full ingest complete for {TEST_REPO}")

# Now search it
from embedder import embed_texts
from qdrant_store import search

query = "how to add a command line argument"
q_vec = embed_texts([query])[0]
results = search(COLLECTION, q_vec, top_k=3)

print(f"\nQuery: '{query}'")
for r in results:
    print(f"\n  [{r['score']:.3f}] {r['file_path']} → {r['name']}")
    print(f"  {r['text'][:150]}")
