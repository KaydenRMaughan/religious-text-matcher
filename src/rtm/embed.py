"""Sentence-embedding encoders with an on-disk cache (Claude's: model loading and caching).

Vectors come back raw (not normalized): normalizing is part of the retriever,
``rtm.retrieve.DenseRetriever``. Cached under data/processed/embeddings/ (gitignored),
keyed by model and a hash of the texts, so a changed corpus never reuses stale vectors.

Models (Apache 2.0, downloaded from the Hugging Face hub on first use):
- minilm: all-MiniLM-L6-v2, 384 dimensions, 256-token limit, ~100 s for the corpus on CPU
- mpnet:  all-mpnet-base-v2, 768 dimensions, 384-token limit, ~8-10 min on CPU
"""
from __future__ import annotations

import functools
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rtm.corpus import REPO_ROOT

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
CACHE_DIR = REPO_ROOT / "data" / "processed" / "embeddings"


@functools.cache
def load_model(key: str):
    from sentence_transformers import SentenceTransformer  # slow import; only when needed

    return SentenceTransformer(MODELS[key], device="cpu")


def encode(texts: Sequence[str], key: str, batch_size: int = 64, progress: bool = False) -> np.ndarray:
    """Raw embeddings, shape (len(texts), dim), float32. No cache: for queries."""
    return load_model(key).encode(
        list(texts), batch_size=batch_size, show_progress_bar=progress, convert_to_numpy=True
    )


def embed_corpus(texts: Sequence[str], key: str, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Embeddings for a whole corpus, computed once and cached as .npy."""
    digest = hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"{key}-{digest}.npy"
    if path.exists():
        return np.load(path)
    vectors = encode(texts, key, progress=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
