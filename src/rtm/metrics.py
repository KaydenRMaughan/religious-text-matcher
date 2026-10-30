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
from math import log2
from collections.abc import Collection, Mapping, Sequence

def _validate(ranked: Sequence[str], relevant: Collection[str], k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        raise ValueError("relevant set is empty")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked list contains duplicate ids")

def recall_at_k(
    ranked: Sequence[str], relevant: Collection[str], k: int, capped: bool = False
) -> float:
    """Share of the relevant set found in the top k.

        uncapped:  |R ∩ L[1..k]| / |R|
        capped:    |R ∩ L[1..k]| / min(|R|, k)

    The capped form scores a perfect top k as 1.0 even when |R| > k; the README
    reports both (decision 5).
    """
    _validate(ranked, relevant, k)
    ranked_top_k = ranked[:k]
    hits = len(set(ranked_top_k) & set(relevant))
    if capped:
        return hits / min(len(relevant), k)
    else:
        return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str], k: int = 10) -> float:
    """1 / rank of the first relevant id within the top k; 0.0 if none.

    MRR@k is the mean of this over queries.
    """
    _validate(ranked, relevant, k)
    ranked_top_k = ranked[:k]
    for rank, doc_id in enumerate(ranked_top_k, start=1):
        if doc_id in relevant: 
            return 1 / rank 
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalized discounted cumulative gain at k.

        DCG@k  = Σ_{i=1..k} gain(L[i]) / log2(i + 1)
        IDCG@k = DCG@k of the ideal list: the k largest gains, in descending order
        nDCG@k = DCG@k / IDCG@k

    ``gains`` maps each relevant id to a positive gain; ids not in it have gain 0.
    Binary relevance passes 1.0 for every relevant id; graded passes log2(1 + votes)
    (decision 2). An empty mapping, or a gain <= 0, raises ValueError.
    """
    _validate(ranked, gains.keys(), k)
    if not gains:
        raise ValueError("gains mapping is empty")
    if any(g <= 0 for g in gains.values()):
        raise ValueError("gains mapping contains non-positive gain")
    ranked_top_k = ranked[:k]
    dcg = sum(gains.get(id, 0) / (log2(i + 1)) for i, id in enumerate(ranked_top_k, start=1))
    ideal_gains = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / (log2(i + 1)) for i, g in enumerate(ideal_gains, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg