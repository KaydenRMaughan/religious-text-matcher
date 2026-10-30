"""Per-query ranking metrics. The evaluation harness averages them over queries.

Spec written 2026-10-30; bodies are Kayden's to type. Tests: tests/test_metrics.py.

Shared conventions:
- ``ranked`` is the retriever's output, best first. It may be shorter than k (a
  small candidate pool); a duplicate id raises ValueError.
- An empty relevant set, or k < 1, raises ValueError: such a query shouldn't
  reach the metrics.
- Ranks are 1-based.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence


def recall_at_k(
    ranked: Sequence[str], relevant: Collection[str], k: int, capped: bool = False
) -> float:
    """Share of the relevant set found in the top k.

        uncapped:  |R ∩ L[1..k]| / |R|
        capped:    |R ∩ L[1..k]| / min(|R|, k)

    The capped form scores a perfect top k as 1.0 even when |R| > k; the README
    reports both (decision 5).
    """
    raise NotImplementedError


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str], k: int = 10) -> float:
    """1 / rank of the first relevant id within the top k; 0.0 if none.

    MRR@k is the mean of this over queries.
    """
    raise NotImplementedError


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalized discounted cumulative gain at k.

        DCG@k  = Σ_{i=1..k} gain(L[i]) / log2(i + 1)
        IDCG@k = DCG@k of the ideal list: the k largest gains, in descending order
        nDCG@k = DCG@k / IDCG@k

    ``gains`` maps each relevant id to a positive gain; ids not in it have gain 0.
    Binary relevance passes 1.0 for every relevant id; graded passes log2(1 + votes)
    (decision 2). An empty mapping, or a gain <= 0, raises ValueError.
    """
    raise NotImplementedError
