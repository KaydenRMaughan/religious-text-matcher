"""Retrievers: a tokenizer and a BM25 baseline (embedding retrievers come later).

Spec written 2026-10-30; bodies are Kayden's to type. Tests: tests/test_retrieve.py.
Decisions behind the spec: docs/decisions.md, 2026-10-30.
"""
from __future__ import annotations

import numpy as np
import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Split text into BM25 terms.

    Rules, in order:
    1. Normalize the curly apostrophe U+2019 to a straight ``'`` (the Gutenberg texts
       use only U+2019; typed queries use ``'``).
    2. Lowercase.
    3. Tokens are runs of letters, optionally joined by one internal apostrophe:
       ``father's`` is one token. Everything else separates, so ``god-ward`` gives
       ``god``, ``ward``. No stopword removal: IDF already down-weights ``thee``.
    4. If ``stem``:
       a. Map KJV verb endings to their modern form first: ``loveth``/``lovest`` must
          stem the same as ``loves``, ``cometh`` as ``comes``. Snowball alone leaves
          every -eth/-est form untouched.
       b. Then apply the Snowball English stemmer (``_STEMMER.stemWord``).

    Rule 4a must not touch words that only *end* in -eth/-est: ``priest``, ``rest``,
    ``west``, ``harvest``, ``Seth``, ``teeth``, ``Nazareth`` (``priest`` alone occurs
    in 570 verses). How to tell them apart is yours: an exception list, a minimum
    stem length, or a check that the modern form exists in the corpus vocabulary.
    Irregular forms (``hath``, ``doth``, ``saith``) are out of scope.
    """
    raise NotImplementedError


class BM25:
    """Okapi BM25 over pre-tokenized documents, with Lucene's IDF.

    For query Q (a token list; repeated tokens count once per occurrence) and document D:

        score(D, Q) = Σ_{q ∈ Q} IDF(q) · f(q, D)·(k1 + 1) / ( f(q, D) + k1·(1 − b + b·|D|/avgdl) )

        IDF(q) = ln( 1 + (N − n(q) + 0.5) / (n(q) + 0.5) )

    f(q, D) is the count of q in D, |D| the token count of D, avgdl the mean |D|,
    N the number of documents, n(q) the number of documents containing q.

    Why Lucene's IDF rather than rank-bm25's Okapi default: Okapi's
    ln((N − n + 0.5)/(n + 0.5)) turns negative for any term in more than half the
    documents, and ``and`` is in 79% of verses. rank-bm25 patches that with an
    ad hoc floor (0.25 × mean IDF); Lucene's form stays positive with no floor.

    A query token not seen in any document contributes 0. Scoring should be
    vectorized over documents (NumPy): the evaluation runs thousands of queries
    against 37,706 verses.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        raise NotImplementedError

    def idf(self, term: str) -> float:
        """IDF of ``term`` as defined above; 0.0 for a term in no document."""
        raise NotImplementedError

    def scores(self, query: list[str]) -> np.ndarray:
        """BM25 score of every document for ``query``, shape (N,), in ``docs`` order."""
        raise NotImplementedError
