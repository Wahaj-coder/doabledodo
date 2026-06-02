"""
describer.py
------------
LLM-powered descriptions for RAG chunks.

Strategy (updated):
  - Group chunks by FILE — all chunks from same file in ONE LLM call
  - Token-safe: if a file's chunks exceed MAX_TOKENS_PER_BATCH, split into
    multiple calls (still far fewer than 1-per-chunk)
  - Each call returns CHUNK_1/CHUNK_2/... blocks → parsed back individually
  - Fallback: if LLM garbles a slot → rule-based for that chunk only
  - Sequential (no ThreadPoolExecutor) — on 1-CPU Ollama parallelism is useless

Why file-grouping works even on qwen2.5:1.5b:
  - All chunks share the same file context → model doesn't get confused
  - Fewer total LLM calls → dramatically faster
  - Worst case a slot is garbled → rule-based fallback, not a crash

Env vars:
  OLLAMA_BASE_URL         default: http://localhost:11434
  DESCRIBE_MODEL          default: qwen2.5:1.5b
  RERANKER_MODEL          default: qwen2.5:1.5b
  DESCRIBE_TIMEOUT        default: 60
  DESCRIBE_SKIP_EXISTING  default: true
  USE_LLM_DESCRIPTIONS    default: true
  MAX_TOKENS_PER_BATCH    default: 3000  (safe limit for 1.5b; ~750 chars/token)
  CHARS_PER_TOKEN         default: 4
"""

import os
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
DESCRIBE_MODEL       = os.getenv("DESCRIBE_MODEL",         "qwen2.5:1.5b")
RERANKER_MODEL       = os.getenv("RERANKER_MODEL",         "qwen2.5:1.5b")
DESCRIBE_TIMEOUT     = int(os.getenv("DESCRIBE_TIMEOUT",   "90"))
SKIP_EXISTING        = os.getenv("DESCRIBE_SKIP_EXISTING", "true").lower() == "true"
USE_LLM              = os.getenv("USE_LLM_DESCRIPTIONS",   "true").lower() == "true"
MAX_TOKENS_PER_BATCH = int(os.getenv("MAX_TOKENS_PER_BATCH", "3000"))
CHARS_PER_TOKEN      = int(os.getenv("CHARS_PER_TOKEN",      "4"))


# ─────────────────────────────────────────────────────────────────────────────
# Token estimator
# ─────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


# ─────────────────────────────────────────────────────────────────────────────
# Build per-file lookup: file_path → context_text + sibling_names
# ─────────────────────────────────────────────────────────────────────────────

def _build_file_index(chunks: List[Dict[str, Any]]) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}

    for chunk in chunks:
        fp = chunk.get("file_path", "")
        if fp not in index:
            index[fp] = {"context_text": "", "sibling_names": []}

        doc_type = chunk.get("doc_type", "")
        name     = chunk.get("name", "")

        if doc_type.endswith("_context") and "[file_context" in name:
            text = chunk.get("text", "")
            index[fp]["context_text"] = text[:1500] if len(text) > 1500 else text
            for sym in chunk.get("symbol_names", []):
                if sym not in index[fp]["sibling_names"]:
                    index[fp]["sibling_names"].append(sym)

        elif doc_type.endswith("_context") and "[context]" in name:
            for sym in chunk.get("symbol_names", []):
                if sym not in index[fp]["sibling_names"]:
                    index[fp]["sibling_names"].append(sym)

        elif name and not name.startswith("["):
            short = name.split(".")[-1]
            if short and short not in index[fp]["sibling_names"]:
                index[fp]["sibling_names"].append(short)

    return index


# ─────────────────────────────────────────────────────────────────────────────
# Filename extraction — always from metadata, never from LLM
# ─────────────────────────────────────────────────────────────────────────────

def _extract_filename(chunk: Dict[str, Any]) -> str:
    fp = chunk.get("file_path", "")
    if fp:
        return Path(fp).name
    name = chunk.get("name", "")
    if "/" in name or "\\" in name:
        return Path(name).name
    return "unknown_file"


# ─────────────────────────────────────────────────────────────────────────────
# System prompt for batched output
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SYSTEM = """You write search index entries for code chunks used in a RAG system.
You will receive multiple numbered chunks from the SAME file.
For EACH chunk output a block in exactly this format:

CHUNK_1:
SUMMARY: <one sentence using concrete keywords from the code>
QUERY1: <natural language question with specific keywords>
QUERY2: <different angle question targeting another aspect>

CHUNK_2:
SUMMARY: ...
QUERY1: ...
QUERY2: ...

Hard rules:
- Output ALL chunks. Do not skip any.
- NEVER start SUMMARY with "This function", "This code", "This class", "This method".
- NEVER use vague words: "this", "that", "the function", "the code", "it".
- ALWAYS use actual operation keywords from the code (e.g. "HMAC validation", "Qdrant upsert").
- NEVER mention filename or programming language.
- QUERY1 and QUERY2 must contain specific nouns from the code.
- Output only the CHUNK blocks. No preamble, no explanation."""


# ─────────────────────────────────────────────────────────────────────────────
# Build prompt for a batch of chunks from the same file
# ─────────────────────────────────────────────────────────────────────────────

def _build_batch_prompt(
    batch: List[Dict[str, Any]],
    file_ctx: Dict,
    filename: str,
) -> str:
    ctx_text      = file_ctx.get("context_text", "")
    sibling_names = file_ctx.get("sibling_names", [])

    lines = [f"FILE: {filename}", ""]

    if ctx_text:
        lines += ["=== FILE OVERVIEW ===", ctx_text, ""]

    if sibling_names:
        lines += [f"SYMBOLS IN FILE: {', '.join(sibling_names[:15])}", ""]

    lines.append("=== CHUNKS TO DESCRIBE ===")
    lines.append("")

    for i, chunk in enumerate(batch, 1):
        name     = chunk.get("name", "")
        doc_type = chunk.get("doc_type", "")
        code     = chunk.get("text", "")
        if len(code) > 800:
            code = code[:800] + "\n... (truncated)"

        def _short(cid: str) -> str:
            parts = cid.split("::")
            return parts[-2] if len(parts) >= 2 else cid

        calls     = [_short(c) for c in chunk.get("calls",     []) if c][:4]
        called_by = [_short(c) for c in chunk.get("called_by", []) if c][:4]

        lines.append(f"--- CHUNK_{i} ---")
        lines.append(f"NAME: {name}  TYPE: {doc_type}")
        if calls:
            lines.append(f"CALLS: {', '.join(calls)}")
        if called_by:
            lines.append(f"CALLED BY: {', '.join(called_by)}")
        lines.append("CODE:")
        lines.append(code)
        lines.append("")

    # Hard format enforcement — repeated at end of prompt for small models
    lines += [
        f"IMPORTANT: There are exactly {len(batch)} chunks above.",
        f"You MUST output exactly {len(batch)} blocks.",
        f"Start immediately with CHUNK_1: — no preamble, no explanation.",
        f"Every block MUST have SUMMARY:, QUERY1:, QUERY2: on separate lines.",
        f"Example of correct output:",
        f"CHUNK_1:",
        f"SUMMARY: <one sentence>",
        f"QUERY1: <question>",
        f"QUERY2: <question>",
        f"CHUNK_2:",
        f"SUMMARY: ...",
        f"QUERY1: ...",
        f"QUERY2: ...",
        f"Now write all {len(batch)} blocks:",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Split file chunks into token-safe batches
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_batches(
    chunks: List[Dict[str, Any]],
    file_ctx: Dict,
    filename: str,
) -> List[List[Dict[str, Any]]]:
    """
    Splits chunks into sub-batches so each prompt stays under MAX_TOKENS_PER_BATCH.
    File overview is included in every batch (counted toward token budget).
    """
    # Fixed overhead per batch: file overview + sibling names + headers
    ctx_tokens  = _approx_tokens(file_ctx.get("context_text", ""))
    base_tokens = ctx_tokens + 200  # headers, labels, instructions

    batches: List[List[Dict]] = []
    current: List[Dict]       = []
    current_tokens            = base_tokens

    for chunk in chunks:
        code   = chunk.get("text", "")[:800]
        name   = chunk.get("name", "")
        # tokens for this chunk: code + name + labels overhead (~50 tokens)
        chunk_tokens = _approx_tokens(code) + _approx_tokens(name) + 50

        if current and current_tokens + chunk_tokens > MAX_TOKENS_PER_BATCH:
            batches.append(current)
            current       = []
            current_tokens = base_tokens

        current.append(chunk)
        current_tokens += chunk_tokens

    if current:
        batches.append(current)

    return batches


# ─────────────────────────────────────────────────────────────────────────────
# LLM call helper
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, model: str, system: str, max_tokens: int = 200) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"

    resp = requests.post(url, json={
        "model":  model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "stop": [],
        },
    }, timeout=DESCRIBE_TIMEOUT)

    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Parse single chunk block → (summary, query1, query2)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_chunk_block(block: str) -> Tuple[str, str, str]:
    summary = ""
    query1  = ""
    query2  = ""
    for line in block.splitlines():
        line = line.strip()
        if line.upper().startswith("SUMMARY:"):
            summary = line[len("SUMMARY:"):].strip()
        elif line.upper().startswith("QUERY1:"):
            query1 = line[len("QUERY1:"):].strip()
        elif line.upper().startswith("QUERY2:"):
            query2 = line[len("QUERY2:"):].strip()
    return summary, query1, query2


# ─────────────────────────────────────────────────────────────────────────────
# Parse full batched LLM response → list of (summary, q1, q2)
# Maps by CHUNK_N position, falls back to ("","","") per missing slot
# ─────────────────────────────────────────────────────────────────────────────

def _parse_batch_response(raw: str, count: int) -> List[Tuple[str, str, str]]:
    """
    Splits LLM output on CHUNK_N: markers and parses each block.
    Returns list of length `count` — empty tuple ("","","") for missing/garbled slots.
    """
    results: List[Tuple[str, str, str]] = [("", "", "")] * count

    # Split on CHUNK_N: markers (case-insensitive)
    parts = re.split(r"(?i)CHUNK_(\d+)\s*:", raw)
    # parts = ["preamble", "1", "block1 text", "2", "block2 text", ...]

    i = 1
    while i < len(parts) - 1:
        try:
            idx   = int(parts[i]) - 1   # 0-indexed
            block = parts[i + 1]
            if 0 <= idx < count:
                results[idx] = _parse_chunk_block(block)
        except (ValueError, IndexError):
            pass
        i += 2

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback — metadata only, no LLM
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based(chunk: Dict[str, Any]) -> Tuple[str, str, str]:
    filename = _extract_filename(chunk)
    name     = chunk.get("name", "")
    doc_type = chunk.get("doc_type", "")
    lang     = chunk.get("language", "")

    def _short(cid): return cid.split("::")[-2] if "::" in cid else cid

    calls   = [_short(c) for c in chunk.get("calls",     []) if c][:5]
    symbols = chunk.get("symbol_names", [])[:12]

    stem = filename.lower().replace("_", "").replace("-", "").replace(".py", "")
    purpose_map = {
        "webhook":  "handles incoming webhook events from external services",
        "ingest":   "orchestrates the repo ingestion pipeline",
        "embed":    "generates vector embeddings for text chunks",
        "chunk":    "splits source files into indexable chunks",
        "store":    "manages vector database read and write operations",
        "search":   "performs similarity search over indexed chunks",
        "auth":     "handles authentication and authorization",
        "model":    "defines data models and schemas",
        "route":    "defines API routes and request handlers",
        "config":   "stores or loads configuration settings",
        "test":     "contains automated test cases",
        "util":     "provides shared utility functions",
        "order":    "manages order creation and processing",
        "payment":  "handles payment processing and transactions",
        "user":     "manages user accounts and profiles",
        "notif":    "sends notifications or alerts",
        "task":     "manages background or async task execution",
        "describ":  "generates LLM-powered descriptions for code chunks",
        "qdrant":   "manages Qdrant vector store read and write operations",
    }
    file_purpose = next((v for k, v in purpose_map.items() if k in stem), f"{lang} source file")
    short_name   = name.split(".")[-1] if name and not name.startswith("[") else doc_type.replace("_", " ")

    if "context" in doc_type and symbols:
        summary = f"File-level overview of {filename} which {file_purpose}, defining {', '.join(symbols[:4])}"
        query1  = f"what does {filename} do"
        query2  = f"which functions are defined in {filename}"
    else:
        calls_str = f", calling {', '.join(calls)}" if calls else ""
        summary   = f"Implements {short_name} in {filename} which {file_purpose}{calls_str}"
        query1    = f"where is {short_name} implemented"
        query2    = f"how does {short_name} work in {filename}"

    return summary, query1, query2


# ─────────────────────────────────────────────────────────────────────────────
# Build description string
# ─────────────────────────────────────────────────────────────────────────────

def _build_description(filename: str, summary: str, query1: str, query2: str) -> str:
    return (
        f"FILENAME: {filename}\n"
        f"SUMMARY: {summary}\n"
        f"QUERY1: {query1}\n"
        f"QUERY2: {query2}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Describe one batch of chunks (same file) — one LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _describe_batch(
    batch: List[Dict[str, Any]],
    file_ctx: Dict,
    filename: str,
) -> None:
    """
    Calls LLM once for the whole batch, parses per-chunk results,
    assigns chunk["description"] on each chunk.
    Falls back to rule-based per-chunk if LLM slot is garbled — no retry.
    Mutates chunks in-place.
    """
    # max_tokens: ~80 tokens output per chunk (SUMMARY+QUERY1+QUERY2 + labels)
    max_tokens = len(batch) * 80

    prompt = _build_batch_prompt(batch, file_ctx, filename)
    raw    = _ollama_generate(prompt, DESCRIBE_MODEL, BATCH_SYSTEM, max_tokens=max_tokens)

    parsed = _parse_batch_response(raw, len(batch))

    # Count garbled slots before assigning
    missing = [i for i, (s, q1, q2) in enumerate(parsed) if not s or not q1 or not q2]
    if missing:
        print(f"[describer] {len(missing)}/{len(batch)} slots garbled in '{filename}' — rule-based for those slots")
        for i in missing:
            parsed[i] = _rule_based(batch[i])

    for chunk, (s, q1, q2) in zip(batch, parsed):
        fn = _extract_filename(chunk)
        chunk["description"] = _build_description(fn, s, q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# Public: describe_chunks — file-grouped, token-safe batching
# ─────────────────────────────────────────────────────────────────────────────

def describe_chunks(
    chunks: List[Dict[str, Any]],
    use_llm: bool = USE_LLM,
) -> List[Dict[str, Any]]:
    """
    Adds chunk["description"] to every chunk.

    Groups chunks by file → one LLM call per file (or per token-safe sub-batch).
    This reduces 200 LLM calls → ~20-30 calls (one per file), dramatically
    faster on a 1-CPU machine where parallelism gives no benefit.

    If a file has too many chunks to fit in MAX_TOKENS_PER_BATCH, it is
    automatically split into multiple calls — still far fewer than 1-per-chunk.

    Fallback: if LLM garbles any individual chunk slot → rule-based for that slot.
    """
    if not chunks:
        return chunks

    to_describe = chunks if not SKIP_EXISTING else [
        c for c in chunks if not c.get("description")
    ]

    if not to_describe:
        print("[describer] All chunks already have descriptions. Skipping.")
        return chunks

    if not use_llm:
        for chunk in to_describe:
            fn        = _extract_filename(chunk)
            s, q1, q2 = _rule_based(chunk)
            chunk["description"] = _build_description(fn, s, q1, q2)
        print(f"[describer] ✅ {len(to_describe)} rule-based descriptions done.")
        return chunks

    # Build file index once
    file_index = _build_file_index(chunks)

    # Group chunks by file
    by_file: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in to_describe:
        by_file[chunk.get("file_path", "")].append(chunk)

    total      = len(to_describe)
    done       = 0
    t0         = time.time()
    total_calls = 0

    print(f"[describer] Describing {total} chunks across {len(by_file)} files — model={DESCRIBE_MODEL}")

    for fp, file_chunks in by_file.items():
        file_ctx = file_index.get(fp, {"context_text": "", "sibling_names": []})
        filename = Path(fp).name if fp else "unknown_file"

        # Split into token-safe batches
        batches = _split_into_batches(file_chunks, file_ctx, filename)

        for batch in batches:
            total_calls += 1
            try:
                _describe_batch(batch, file_ctx, filename)
            except requests.exceptions.ConnectionError:
                print(f"\n[describer] ❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}.")
                print("[describer] Switching all remaining chunks to rule-based.")
                # Fallback ALL remaining undescribed chunks
                for c in to_describe:
                    if not c.get("description"):
                        fn        = _extract_filename(c)
                        s, q1, q2 = _rule_based(c)
                        c["description"] = _build_description(fn, s, q1, q2)
                elapsed = time.time() - t0
                print(f"[describer] ✅ Done (rule-based fallback) in {elapsed:.1f}s")
                return chunks
            except Exception as e:
                print(f"[describer] Batch error for {filename}: {e} — rule-based fallback for this batch")
                for chunk in batch:
                    fn        = _extract_filename(chunk)
                    s, q1, q2 = _rule_based(chunk)
                    chunk["description"] = _build_description(fn, s, q1, q2)

            done += len(batch)
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 1
            eta     = (total - done) / rate
            print(
                f"[describer] {done}/{total} chunks  "
                f"{total_calls} LLM calls  "
                f"{elapsed:.1f}s elapsed  ~{eta:.0f}s remaining",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"[describer] ✅ {total} descriptions done in {elapsed:.1f}s  ({total_calls} LLM calls total)")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public: enrich_embed_text
# ─────────────────────────────────────────────────────────────────────────────

def enrich_embed_text(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds chunk["embed_text"] = description + original code.
    Call AFTER describe_chunks(), BEFORE embed_chunks().
    chunk["text"] is preserved untouched for UI display.
    """
    for chunk in chunks:
        desc     = chunk.get("description", "")
        original = chunk.get("text", "")
        chunk["embed_text"] = f"{desc}\n\n{original}".strip() if desc else original
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# patch_embedder — zero edits to embedder.py required
# ─────────────────────────────────────────────────────────────────────────────

def patch_embedder():
    """
    Wraps embedder.embed_chunks to use chunk["embed_text"] when present.
    Call ONCE at the top of ingestor.py.
    """
    import embedder
    _orig = embedder.embed_chunks

    def _patched(chunks, **kwargs):
        swapped = []
        for c in chunks:
            if "embed_text" in c:
                c["_orig_text"] = c["text"]
                c["text"]       = c["embed_text"]
                swapped.append(c)
        result = _orig(chunks, **kwargs)
        for c in swapped:
            c["text"] = c.pop("_orig_text")
            c.pop("embed_text", None)
        return result

    embedder.embed_chunks = _patched
    print("[describer] Patched embedder.embed_chunks → uses embed_text when present.")


# ─────────────────────────────────────────────────────────────────────────────
# Reranker — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

_RERANKER_SYSTEM = """You are a code search relevance judge. A developer searched for something and got a code chunk back. Score how relevant that chunk is to their query.

Score from 0 to 10:
  10 = exactly what they are looking for
  7-9 = highly relevant, directly addresses the query
  4-6 = partially relevant, related but not the best match
  1-3 = loosely related
  0 = not relevant

Output ONLY a single integer (0-10). No explanation. No other text."""


def _score_one_result(query: str, result: Dict[str, Any]) -> float:
    filename = _extract_filename(result)
    name     = result.get("name", "")
    desc     = result.get("description", "")
    code     = result.get("text", "")[:600]

    prompt = (
        f'Developer query: "{query}"\n\n'
        f"Code chunk:\n"
        f"  File: {filename}\n"
        f"  Function/block: {name}\n"
        f"  Description: {desc}\n"
        f"  Code preview:\n{code}\n\n"
        f"Score (0-10):"
    )

    try:
        raw   = _ollama_generate(prompt, RERANKER_MODEL, _RERANKER_SYSTEM, max_tokens=4)
        m     = re.search(r"\d+", raw)
        score = float(m.group()) if m else 0.0
        return min(10.0, max(0.0, score)) / 10.0
    except Exception:
        return result.get("score", 0.0)


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Re-scores search results using LLM relevance judgment.
    Sequential — one call per result (reranking is already a small set, ~20 results).
    """
    if not results:
        return results

    print(f"[reranker] Scoring {len(results)} candidates for: '{query[:70]}'")
    t0 = time.time()

    for r in results:
        try:
            r["rerank_score"] = _score_one_result(query, r)
        except Exception:
            r["rerank_score"] = r.get("score", 0.0)

    ranked  = sorted(results, key=lambda r: r.get("rerank_score", 0.0), reverse=True)
    elapsed = time.time() - t0

    print(f"[reranker] ✅ Done in {elapsed:.1f}s")
    if ranked:
        best = ranked[0]
        fn   = _extract_filename(best)
        print(f"[reranker] Best: '{best.get('name','?')}' in {fn} "
              f"(rerank={best.get('rerank_score',0):.2f}, cosine={best.get('score',0):.2f})")

    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm",  action="store_true", help="Use rule-based fallback")
    ap.add_argument("--rerank",  action="store_true", help="Also test reranker")
    args = ap.parse_args()

    test_chunks = [
        {
            "file_path":    "webhook_server.py",
            "doc_type":     "code_context",
            "name":         "[file_context]",
            "language":     "python",
            "calls":        [],
            "called_by":    [],
            "symbol_names": ["github_webhook", "bitbucket_webhook", "manual_trigger",
                             "verify_github_signature", "health"],
            "text": (
                "import os, hmac, hashlib, json\n"
                "from fastapi import FastAPI, Request, HTTPException\n"
                "WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '').encode()\n"
                "app = FastAPI(title='RAG Webhook Server')\n"
            ),
        },
        {
            "file_path":    "webhook_server.py",
            "doc_type":     "code",
            "name":         "verify_github_signature",
            "language":     "python",
            "calls":        [],
            "called_by":    [],
            "symbol_names": [],
            "text": (
                "def verify_github_signature(body: bytes, sig_header: str) -> bool:\n"
                "    if not sig_header or not sig_header.startswith('sha256='):\n"
                "        return False\n"
                "    expected = 'sha256=' + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()\n"
                "    return hmac.compare_digest(expected, sig_header)\n"
            ),
        },
    ]

    print("=" * 60)
    described = describe_chunks(test_chunks, use_llm=not args.no_llm)
    enriched  = enrich_embed_text(described)

    for c in enriched:
        print(f"\n{'─'*50}")
        print(f"CHUNK : {c['name']}  ({c['file_path']})")
        print(f"\nDESCRIPTION:\n{c['description']}")
        print(f"\nEMBED TEXT preview:\n{c['embed_text'][:300]}...")

    if args.rerank:
        print("\n" + "=" * 60)
        fake  = [{**c, "score": 0.75} for c in described]
        query = "where are old chunks deleted when a file changes"
        ranked = rerank(query, fake, top_k=2)
        print(f"\nTop 2 for: '{query}'")
        for r in ranked:
            print(f"  rerank={r['rerank_score']:.2f}  cosine={r['score']:.2f}  {r['name']}")