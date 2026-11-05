"""Shared setup for the evaluation scripts: corpus, labels, query samples, retrievers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rtm.corpus import RAW_DIR, REPO_ROOT, load_cross_references, load_scripture
from rtm.evaluate import PRIMARY_CUT, VOTE_CUTS, build_qrels, split_queries

SEED = 20261030
N_DEV = N_TEST = 2000
RESULTS = REPO_ROOT / "results"


@dataclass
class EvalSetup:
    verses: pd.DataFrame
    qrels: dict
    judged: set
    dev: list
    test: list


def load_setup() -> EvalSetup:
    verses = load_scripture()
    xrefs = load_cross_references(RAW_DIR / "cross_references.txt", verses["id"].tolist())
    judged = set(verses.loc[verses["source"] != "Book of Mormon", "id"])
    qrels = {cut: build_qrels(xrefs, cut) for cut in VOTE_CUTS}
    stratum = verses.set_index("id")["source"].to_dict()
    dev, test = split_queries(qrels[PRIMARY_CUT].keys(), stratum, N_DEV, N_TEST, SEED)
    return EvalSetup(verses, qrels, judged, dev, test)


def dense_score_fn(verses: pd.DataFrame, key: str):
    """Score function for a corpus-embedding model. Evaluation queries are corpus
    verses, so a query's vector is looked up from the cached corpus embeddings
    rather than re-encoded (same model, same text, same vector)."""
    from rtm.embed import embed_corpus
    from rtm.retrieve import DenseRetriever

    vectors = embed_corpus(verses["text"].tolist(), key)
    row = {}
    for i, text in enumerate(verses["text"]):
        row.setdefault(text, i)
    retriever = DenseRetriever(vectors)
    return lambda q: retriever.scores(vectors[row[q]])


def bm25_score_fn(verses: pd.DataFrame, k1: float, b: float, tokens=None):
    from rtm.retrieve import BM25, tokenize

    tokens = tokens or [tokenize(t, stem=True) for t in verses["text"]]
    bm25 = BM25(tokens, k1=k1, b=b)
    return lambda q: bm25.scores(tokenize(q, stem=True))


def hybrid_score_fn(verses: pd.DataFrame, k1: float, b: float):
    """BM25 + mpnet, fused by reciprocal rank (rtm.retrieve.rrf_scores)."""
    from rtm.retrieve import rrf_scores

    bm25, mpnet = bm25_score_fn(verses, k1, b), dense_score_fn(verses, "mpnet")
    return lambda q: rrf_scores([bm25(q), mpnet(q)])


def as_float(x) -> float:
    return float(np.asarray(x))
