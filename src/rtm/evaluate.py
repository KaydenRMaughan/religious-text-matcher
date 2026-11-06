"""Evaluation protocol: labels, query sets, per-query scoring, and the batch runner.

Spec written 2026-10-30. The protocol functions are Kayden's to type; ``run`` and
``summarize`` (the batch runner) are Claude's. Tests: tests/test_evaluate.py.
Protocol: docs/how-it-works.md, "Evaluation protocol".

Flow:
    cross-references --build_qrels--> {query: {target: votes}}   (one per vote cut)
    eligible queries --split_queries--> dev, test
    retriever scores --run--> ranked ids --evaluate_query--> per-query metrics
    per-query metrics --summarize--> means per cut;  paired_bootstrap compares models
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from rtm.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

VOTE_CUTS = (3, 5, 10)
PRIMARY_CUT = 5
METRICS = ("recall", "recall_capped", "mrr", "ndcg", "ndcg_graded", "unjudged_share")


# --- Kayden: the protocol -----------------------------------------------------------

def build_qrels(xrefs: pd.DataFrame, min_votes: int) -> dict[str, dict[str, int]]:
    """Relevance labels ("qrels") at one vote cut: {query id: {relevant id: votes}}.

    ``xrefs`` has columns ``from_id``, ``to_id``, ``votes`` (one row per verse pair,
    no duplicate pairs). Keep links with ``votes >= min_votes``, drop self-links
    (``from_id == to_id``, decision 4), and leave out any query left with no links.
    """
    xrefs = xrefs[(xrefs["votes"] >= min_votes) & (xrefs["from_id"] != xrefs["to_id"])]
    qrels = {}
    for f, t, v in xrefs[["from_id", "to_id", "votes"]].itertuples(index=False):
        qrels.setdefault(f, {})[t] = int(v)
    return qrels

def split_queries(
    eligible: Collection[str],
    stratum: Mapping[str, str],
    n_dev: int,
    n_test: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Two disjoint, seeded samples of query ids, stratified by ``stratum`` (decision 3).

    ``stratum`` maps each id to its group (here the testament). Proportional
    allocation: for a sample of size n, a group with share p of ``eligible`` gets
    round(n · p) ids, and the last group in sorted name order takes the remainder,
    so sizes are exact. Draw dev first, then test from what's left.

    Deterministic: the same ``eligible`` set and ``seed`` give the same samples
    whatever order ``eligible`` arrives in (sort before sampling). Use
    ``np.random.default_rng(seed)``. Return both lists sorted. Raise ValueError if
    n_dev + n_test exceeds len(eligible).
    """
    def _draw(pool: list[str], n:int, stratum, rng) -> list[str]:
        """Draw n from pool, stratified by stratum, with proportional allocation."""
        if n > len(pool):
            raise ValueError(f"n ({n}) exceeds pool size ({len(pool)})")
        # Group ids by stratum
        groups = {}
        for id_ in pool:
            groups.setdefault(stratum[id_], []).append(id_)

        # Determine how many to draw from each group
        total = len(pool)
        allocations = {g: round(n * len(ids) / total) for g, ids in groups.items()}
        # Adjust the last group to take any remainder
        last_group = sorted(groups.keys())[-1]
        allocations[last_group] += n - sum(allocations.values())
        # Draw from each group
        drawn = []
        for g, ids in groups.items():
            rng.shuffle(ids)
            drawn.extend(ids[:allocations[g]])
        return sorted(drawn)
    
    pool = sorted(eligible)
    if n_dev + n_test > len(pool):
        raise ValueError(f"n_dev + n_test ({n_dev + n_test}) exceeds eligible ({len(pool)})")
    rng = np.random.default_rng(seed)
    dev = _draw(pool, n_dev, stratum, rng)
    dev_set = set(dev)
    test = _draw([id_ for id_ in pool if id_ not in dev_set], n_test, stratum, rng)
    return sorted(dev), sorted(test)

def filter_ranking(ranked: Sequence[str], query_id: str, judged: Collection[str]) -> list[str]:
    """The ranking the Bible metrics see: drop the query itself and unjudged ids.

    Unjudged ids are those not in ``judged`` (the Book of Mormon, decision 8a:
    condensed lists, Sakai 2007). Order is preserved.
    """
    # keep d if it's not the query itself and it's judged
    return [d for d in ranked if d != query_id and d in judged]


def gains(qrel: Mapping[str, int], graded: bool) -> dict[str, float]:
    """Gain per relevant id for nDCG: 1.0 if binary, log2(1 + votes) if graded (decision 2)."""
    if graded:
        return {d: np.log2(1 + v) for d, v in qrel.items()}
    return {d: 1.0 for d in qrel}


def evaluate_query(
    ranked: Sequence[str],
    query_id: str,
    qrel: Mapping[str, int],
    judged: Collection[str],
    k: int = 10,
) -> dict[str, float]:
    """All metrics for one query, keyed by METRICS.

    ``ranked`` is the retriever's raw ranking (Bible and Book of Mormon mixed, query
    included). Scoring metrics use the condensed list from ``filter_ranking``:
    recall@k uncapped and capped, MRR@k, nDCG@k binary and graded.

    ``unjudged_share`` is diagnostic: the share of the raw top k, after removing only
    the query itself, that is unjudged. It shows how much the Book of Mormon competes
    for the top slots.
    """
    condensed = filter_ranking(ranked, query_id, judged)
    raw = [d for d in ranked if d != query_id][:k]
    return {
        "recall": recall_at_k(condensed, qrel, k),
        "recall_capped": recall_at_k(condensed, qrel, k, capped=True),
        "mrr": reciprocal_rank(condensed, qrel, k),
        "ndcg": ndcg_at_k(condensed, gains(qrel, graded=False), k),
        "ndcg_graded": ndcg_at_k(condensed, gains(qrel, graded=True), k),
        "unjudged_share": 1 - len([d for d in raw if d in judged]) / len(raw) if raw else 0.0,
    }


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a − b over queries, with a 95% percentile bootstrap interval.

    ``a`` and ``b`` are one metric for two models on the same queries, in the same
    order. Resample *query indexes* with replacement (the same indexes for both
    models: that's what makes it paired), take mean(a[i] − b[i]) per resample, and
    return (observed mean difference, 2.5th percentile, 97.5th percentile). Use
    ``np.random.default_rng(seed)``. Raise ValueError on unequal or zero lengths.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"unequal lengths: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("zero length")
    d = a - b
    rng = np.random.default_rng(seed)
    means = [np.mean(d[rng.choice(len(d), size=len(d), replace=True)]) for _ in range(n_boot)]
    return np.mean(d), np.percentile(means, 2.5), np.percentile(means, 97.5)


def grid_search(
    make_score_fn: Callable[..., Callable[[str], np.ndarray]],
    grid: Mapping[str, Sequence],
    query_ids: Sequence[str],
    verses: pd.DataFrame,
    qrels: Mapping[int, Mapping[str, Mapping[str, int]]],
    judged: Collection[str],
    metric: str = "ndcg",
    cut: int = PRIMARY_CUT,
    **run_kwargs,
) -> pd.DataFrame:
    """Score every combination of ``grid`` on the dev queries; best first.

    ``grid`` maps parameter names to candidate values, e.g. {"k1": [0.9, 1.2],
    "b": [0.5, 0.75]}: every combination (the Cartesian product) is tried.
    ``make_score_fn(**params)`` builds a retriever's score function for one
    combination; ``run`` it at the single vote cut ``cut`` and average ``metric``.

    Return one row per combination: a column per parameter, then ``metric`` (the mean)
    and ``n_queries``. Sort by ``metric`` descending; ties keep grid order (a stable
    sort), so the first row is reproducible. Raise ValueError for an unknown metric.

    Dev queries only: tuning on test would leak into the reported numbers (decision 3).
    ``run_kwargs`` pass through to ``run`` (``query_text``, ``exclude``; added by
    Claude for the private and chapter tasks).
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        results = run(make_score_fn(**params), query_ids, verses, {cut: qrels[cut]}, judged,
                      **run_kwargs)
        rows.append({**params, metric: results[metric].mean(), "n_queries": len(results)})
    table = pd.DataFrame(rows, columns=[*names, metric, "n_queries"])
    return table.sort_values(metric, ascending=False, kind="stable").reset_index(drop=True)


# --- Claude: the batch runner ---------------------------------------------------------

TOP_M = 200  # raw ranks kept per query: ample for 10 judged hits after condensing


def _top_m(scores: np.ndarray, m: int) -> np.ndarray:
    """Indexes of the m highest scores, best first, ties in corpus order.

    Same result as a full stable argsort cut to m, ~15x faster on 37k verses: keep
    everything at or above the m-th largest score, then sort only those.
    """
    if m >= len(scores):
        return np.argsort(-scores, kind="stable")
    cutoff = np.partition(scores, -m)[-m]
    candidates = np.flatnonzero(scores >= cutoff)
    return candidates[np.argsort(-scores[candidates], kind="stable")][:m]


def run(
    score_fn: Callable[[str], np.ndarray],
    query_ids: Sequence[str],
    verses: pd.DataFrame,
    qrels: Mapping[int, Mapping[str, Mapping[str, int]]],
    judged: Collection[str],
    k: int = 10,
    query_text: Mapping[str, str] | None = None,
    exclude: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame:
    """Score every query at every vote cut. One row per (query, cut) with METRICS.

    ``score_fn`` maps query text to one score per row of ``verses`` (the documents).
    ``qrels`` maps each vote cut to its labels; a query with no labels at a cut is
    skipped there. Ties break by corpus order, so runs are reproducible.

    By default queries are rows of ``verses``. ``query_text`` supplies query text from
    elsewhere (talks as queries over verses). ``exclude`` lists, per query, documents to
    drop before scoring (the query verse's own chapter in the chapter task).
    """
    text = query_text if query_text is not None else verses.set_index("id")["text"]
    ids = verses["id"].to_numpy()
    judged = set(judged)
    exclude = exclude or {}
    rows = []
    for qid in query_ids:
        scores = np.asarray(score_fn(text[qid]), dtype=float)
        drop = exclude.get(qid, ())
        ranked = [d for d in ids[_top_m(scores, TOP_M + len(drop))].tolist() if d not in drop]
        for cut, labels in qrels.items():
            if qid in labels:
                rows.append({"query_id": qid, "min_votes": cut,
                             **evaluate_query(ranked, qid, labels[qid], judged, k)})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per vote cut, with the number of queries scored."""
    out = results.groupby("min_votes")[list(METRICS)].mean()
    out.insert(0, "n_queries", results.groupby("min_votes").size())
    return out
