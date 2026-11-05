"""Retrievers: a tokenizer and a BM25 baseline (embedding retrievers come later).

Spec written 2026-10-30; bodies are Kayden's to type. Tests: tests/test_retrieve.py.
Decisions behind the spec: docs/decisions.md, 2026-10-30.
"""
from __future__ import annotations
import collections
import functools
import re

from collections.abc import Sequence

import numpy as np
import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Words of 5+ letters ending in -eth/-est that aren't KJV verb forms. Built 2026-10-30
# from the corpus: every such word with no modern form (base, base+s, base+es, base+ed,
# base+ing) elsewhere in the corpus, minus the rare verbs among them, plus non-verbs
# that do have a spurious match (forest -> for). Superlatives (greatest) are left to
# the rule: mapping them onto their base is a harmless conflation.
_NOT_VERBS = frozenset({
    # nouns, adjectives, ordinals
    "chest", "detest", "dishonest", "forest", "guest", "harvest", "honest", "infest",
    "interest", "manifest", "modest", "priest", "protest", "request", "shibboleth",
    "sibboleth", "teeth", "tempest",
    "twentieth", "thirtieth", "fortieth", "sixtieth", "eightieth", "ninetieth",
    # names
    "alameth", "alemeth", "ashtoreth", "azmaveth", "benzoheth", "bethazmaveth",
    "chinnereth", "dabbasheth", "elisabeth", "hammoleketh", "harosheth", "hazarmaveth",
    "ishbosheth", "japheth", "jerubbesheth", "jetheth", "kirharaseth", "kirhareseth",
    "mephibosheth", "meshullemeth", "mispereth", "nazareth", "peleth", "pibeseth",
    "pochereth", "remeth", "shallecheth", "sophereth", "tanhumeth", "tebeth", "topheth",
    "zoheleth", "zoheth",
})


@functools.cache  # ~14k distinct words vs ~1M tokens: stem each word once
def _stem(token: str) -> str:
    """KJV -eth/-est verb ending -> modern -es, then Snowball (rule 4)."""
    if len(token) > 4 and token.endswith(("eth", "est")) and token not in _NOT_VERBS:
        token = token[:-3] + "es"  # lov-eth -> lov-es, know-est -> know-es
    return _STEMMER.stemWord(token)


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
    tokens = _TOKEN.findall(text.replace("’", "'").lower())
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


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
        """Initialize with pre-tokenized documents, and optional BM25 parameters."""
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.doc_len = np.array([len(doc) for doc in docs], dtype=float)
        self.avgdl = np.mean(self.doc_len)
        self.norm = k1 * (1 - b + b * (self.doc_len / self.avgdl))
        postings = collections.defaultdict(list)
        for i, doc in enumerate(docs):
            for term, count in collections.Counter(doc).items():
                postings[term].append((i, count))
        self.inverted_index = {
            term: (
                np.array([i for i, _ in pairs]),
                np.array([c for _, c in pairs],dtype=float)
            )
            for term, pairs in postings.items()
        }

    def idf(self, term: str) -> float:
        """IDF of ``term`` as defined above; 0.0 for a term in no document."""
        if term not in self.inverted_index:
            return 0.0
        n_q = len(self.inverted_index[term][0])
        return float(np.log(1 + (self.N - n_q + 0.5) / (n_q + 0.5)))

    def scores(self, query: list[str]) -> np.ndarray:
        """BM25 score of every document for ``query``, shape (N,), in ``docs`` order."""
        scores = np.zeros(self.N, dtype=float)
        for term in query:
            if term not in self.inverted_index:
                continue
            idx, f = self.inverted_index[term]
            idf = self.idf(term)
            scores[idx] += idf * f * (self.k1 + 1) / (f + self.norm[idx])
        return scores


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row of ``x`` to unit L2 length, as float64; a 1-D ``x`` is one vector.

        x̂ = x / ‖x‖₂,   ‖x‖₂ = sqrt(Σⱼ xⱼ²)

    A zero vector has no direction: leave it as zeros (it then scores 0 against
    everything) rather than dividing by zero. Doesn't modify ``x``.
    """
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


class DenseRetriever:
    """Exact cosine-similarity search over sentence embeddings.

        cos(q, d) = q·d / (‖q‖ ‖d‖) = q̂·d̂

    Normalize the document vectors once, at construction; then one query against
    all N documents is a single matrix-vector product, D̂ q̂, shape (N,). Scores lie
    in [−1, 1] and don't depend on the query vector's length.
    """

    def __init__(self, doc_vectors: np.ndarray) -> None:
        self.docs = l2_normalize(doc_vectors)  # unit rows, once per model

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of every document to ``query_vector`` (shape (dim,)), shape (N,)."""
        return self.docs @ l2_normalize(query_vector)  # (N, dim) @ (dim,) -> (N,) cosines


def rrf_scores(score_arrays: Sequence[np.ndarray], c: float = 60.0) -> np.ndarray:
    """Reciprocal rank fusion: combine several retrievers' scores over the same documents.

        RRF(d) = Σ_m 1 / (c + rank_m(d))

    rank_m(d) is d's 1-based rank under retriever m (highest score = rank 1), with ties
    broken by corpus order (a stable sort), as in the evaluation runner. Only ranks are
    used, so BM25 scores (unbounded) and cosines ([−1, 1]) combine without calibration.
    c = 60 (Cormack, Clarke & Büttcher, 2009) damps the gap between ranks 1 and 2 so
    that no single retriever dominates. It's fixed, not tuned.

    All arrays must have the same shape (N,); raise ValueError otherwise, or if
    ``score_arrays`` is empty.
    """
    if not score_arrays:
        raise ValueError("score_arrays must be non-empty")
    N = score_arrays[0].shape
    if any(arr.shape != N for arr in score_arrays):
        raise ValueError("all score arrays must have the same shape")
    ranks = np.zeros((len(score_arrays), *N), dtype=int)
    for i, scores in enumerate(score_arrays):
        ranks[i] = np.argsort(np.argsort(-scores, kind="stable")) + 1  # 1-based
    return np.sum(1 / (c + ranks), axis=0)
