"""
bm25_encoder.py
---------------
Lightweight BM25 sparse vector encoder producing Qdrant-compatible
{"indices": [...], "values": [...]} sparse vectors.

No external BM25 library — pure Python with Robertson-Sparck Jones IDF.
Vocab grows during ingest and is persisted to bm25_vocab.json.

Usage:
    enc = BM25Encoder.load_or_create("bm25_vocab.json")
    enc.fit(all_texts)                        # build IDF from corpus
    sparse = enc.encode("some query text")    # encode at search time
    enc.save("bm25_vocab.json")               # persist after ingest
"""

import re, math, json, os
from typing import Dict, List, Optional, Tuple
from collections import Counter

_SPLIT_RE = re.compile(
    r"[^a-zA-Z0-9]+"
    r"|(?<=[a-z])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
)

STOPWORDS = {
    "the","a","an","and","or","is","in","on","at","to","of","for","with",
    "as","by","from","that","this","it","be","are","was","were","has",
    "have","had","not","but","if","then","so","do","does","did","will",
    "would","could","should","can","may","might","i","we","you","he",
    "she","they","def","return","import","class","self","none","true",
    "false","var","let","const","function","new","null","undefined",
}
MIN_TOKEN_LEN = 2


def tokenize(text: str) -> List[str]:
    raw = _SPLIT_RE.sub(" ", text).lower().split()
    return [t for t in raw if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


class BM25Encoder:
    """BM25 sparse encoder compatible with Qdrant sparse vectors."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.vocab: Dict[str, int]  = {}
        self.idf:   Dict[int, float] = {}
        self.avgdl: float = 1.0
        self._next_id = 0

    # ── vocab ────────────────────────────────────────────────────────────────

    def _get_or_add(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self._next_id += 1
        return self.vocab[token]

    def _get(self, token: str) -> Optional[int]:
        return self.vocab.get(token)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "BM25Encoder":
        """Build vocab + IDF from a list of texts. Safe to call incrementally."""
        if not texts:
            return self
        doc_count = len(texts)
        df: Counter = Counter()
        total_len   = 0
        for text in texts:
            toks = tokenize(text)
            total_len += len(toks)
            for tok in set(toks):
                df[tok] += 1
                self._get_or_add(tok)
        self.avgdl = total_len / max(doc_count, 1)
        for token, freq in df.items():
            idx = self.vocab[token]
            self.idf[idx] = math.log(
                1.0 + (doc_count - freq + 0.5) / (freq + 0.5)
            )
        print(f"[bm25] Vocab size: {len(self.vocab)}, avgdl: {self.avgdl:.1f}")
        return self

    # ── encode ───────────────────────────────────────────────────────────────

    def encode(self, text: str, add_new_tokens: bool = False) -> Dict:
        """
        Encode text as BM25 sparse vector.
        Returns {"indices": [...], "values": [...]}
        Set add_new_tokens=True during ingest, False during query.
        """
        tokens = tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}
        dl  = len(tokens)
        tf  = Counter(tokens)
        pairs: List[Tuple[int, float]] = []
        for token, freq in tf.items():
            idx = self._get_or_add(token) if add_new_tokens else self._get(token)
            if idx is None:
                continue
            idf_val = self.idf.get(idx, 1.0)
            tf_val  = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            )
            score = idf_val * tf_val
            if score > 0:
                pairs.append((idx, score))
        if not pairs:
            return {"indices": [], "values": []}
        pairs.sort(key=lambda x: x[0])
        return {
            "indices": [p[0] for p in pairs],
            "values":  [round(p[1], 6) for p in pairs],
        }

    def encode_batch(self, texts: List[str], add_new_tokens: bool = False) -> List[Dict]:
        return [self.encode(t, add_new_tokens=add_new_tokens) for t in texts]

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "k1": self.k1, "b": self.b, "avgdl": self.avgdl,
                "next_id": self._next_id, "vocab": self.vocab,
                "idf": {str(k): v for k, v in self.idf.items()},
            }, f)
        print(f"[bm25] Saved vocab ({len(self.vocab)} tokens) → {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Encoder":
        with open(path) as f:
            data = json.load(f)
        enc = cls(k1=data["k1"], b=data["b"])
        enc.avgdl    = data["avgdl"]
        enc._next_id = data["next_id"]
        enc.vocab    = data["vocab"]
        enc.idf      = {int(k): v for k, v in data["idf"].items()}
        print(f"[bm25] Loaded vocab ({len(enc.vocab)} tokens) ← {path}")
        return enc
    @classmethod
    def load_or_create(cls, path: str) -> "BM25Encoder":
        if os.path.exists(path):
            return cls.load(path)
        print(f"[bm25] No vocab at {path}, starting fresh.")
        return cls()