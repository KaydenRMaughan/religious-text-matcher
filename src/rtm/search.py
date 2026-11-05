"""Search engine behind the app: any query text against a corpus of verses or documents.

Wraps the evaluated retrievers (BM25, MiniLM, mpnet, and their RRF hybrid) for
interactive use:
- long queries (a pasted talk or chapter) are chunked; each document keeps its best
  query passage (max-pool);
- long documents (talks) are chunked too; a document scores as its best passage;
- an optional boost prefers Book of Mormon verses when scores are close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rtm.chunk import chunk_text
from rtm.corpus import REPO_ROOT
from rtm.embed import CACHE_DIR, embed_corpus, encode
from rtm.retrieve import BM25, DenseRetriever, boost, l2_normalize, rrf_scores, tokenize

PRIVATE_CACHE = REPO_ROOT / "data" / "private" / "embeddings"
PUBLIC_SOURCES = ("Old Testament", "New Testament", "Book of Mormon")
METHODS = ("hybrid", "bm25", "mpnet", "minilm")
K1, B = 0.4, 1.0  # tuned on public dev, 2026-11-05
BOOST_DELTA = 0.0  # Book of Mormon margin in hybrid (RRF) units; set from private dev tuning


def unit_vectors(texts: list[str], sources: list[str], key: str) -> np.ndarray:
    """Embeddings for a list of texts, reusing the public verse cache where it applies.

    The public KJV + Book of Mormon verses (in corpus order) hit the cache made for the
    evaluation; anything else is cached under data/private/ (gitignored).
    """
    sources = np.asarray(sources)
    public = np.isin(sources, PUBLIC_SOURCES)
    parts = []
    for mask, cache in ((public, CACHE_DIR), (~public, PRIVATE_CACHE)):
        if mask.any():
            parts.append((mask, embed_corpus([t for t, m in zip(texts, mask) if m], key, cache)))
    dim = parts[0][1].shape[1]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for mask, vecs in parts:
        out[mask] = vecs
    return out


@dataclass
class Index:
    """A searchable corpus. ``docs`` needs ``id``, ``text`` and ``source`` columns."""

    docs: pd.DataFrame
    chunk_docs: bool = False          # True for long documents (talks)
    _dense: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.docs = self.docs.reset_index(drop=True)
        self.bm25 = BM25([tokenize(t, stem=True) for t in self.docs["text"]], k1=K1, b=B)
        self.is_bom = (self.docs["source"] == "Book of Mormon").to_numpy()
        if self.chunk_docs:
            self.passages = [chunk_text(t) or [t] for t in self.docs["text"]]
            self.starts = np.cumsum([0] + [len(p) for p in self.passages])[:-1]

    def _retriever(self, key: str) -> DenseRetriever:
        if key not in self._dense:
            if self.chunk_docs:
                flat = [p for ps in self.passages for p in ps]
                src = [s for s, ps in zip(self.docs["source"], self.passages) for _ in ps]
                self._dense[key] = DenseRetriever(unit_vectors(flat, src, key))
            else:
                self._dense[key] = DenseRetriever(
                    unit_vectors(self.docs["text"].tolist(), self.docs["source"].tolist(), key))
        return self._dense[key]

    def scores(self, query: str, method: str = "hybrid") -> np.ndarray:
        """One score per document."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        if method == "bm25":
            return self.bm25.scores_counted(tokenize(query, stem=True))
        if method == "hybrid":
            return rrf_scores([self.scores(query, "bm25"), self.scores(query, "mpnet")])
        query_vecs = encode(chunk_text(query) or [query], method)
        units = (self._retriever(method).docs @ l2_normalize(query_vecs).T).max(axis=1)
        return np.maximum.reduceat(units, self.starts) if self.chunk_docs else units

    def search(self, query: str, method: str = "hybrid", k: int = 10,
               delta: float = 0.0, sources: list[str] | None = None) -> pd.DataFrame:
        """Top ``k`` documents with their scores. ``delta`` boosts Book of Mormon verses;
        ``sources`` limits results to those sources."""
        s = boost(self.scores(query, method), self.is_bom, delta)
        if sources:
            s = np.where(self.docs["source"].isin(sources).to_numpy(), s, -np.inf)
        top = np.argsort(-s, kind="stable")[:k]
        top = top[np.isfinite(s[top])]
        out = self.docs.iloc[top].copy()
        out.insert(0, "score", s[top])
        out.insert(0, "rank", np.arange(1, len(top) + 1))
        return out.reset_index(drop=True)

