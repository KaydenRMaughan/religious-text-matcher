from math import log2

import numpy as np
import pandas as pd
import pytest

from rtm.evaluate import (
    METRICS, build_qrels, evaluate_query, filter_ranking, gains, paired_bootstrap, run,
    split_queries, summarize,
)

XREFS = pd.DataFrame(
    [
        ("Gen.1.1", "John.1.1", 50),
        ("Gen.1.1", "Heb.11.3", 7),
        ("Gen.1.1", "Ps.33.6", 3),
        ("Gen.1.1", "Gen.1.1", 9),   # self-link
        ("Gen.1.2", "Isa.45.18", 2),
        ("Gen.1.3", "2Cor.4.6", 12),
        ("Gen.1.3", "Gen.1.4", -4),
    ],
    columns=["from_id", "to_id", "votes"],
)


# --- build_qrels ----------------------------------------------------------------------

def test_qrels_apply_vote_cut_and_keep_votes():
    assert build_qrels(XREFS, 5) == {
        "Gen.1.1": {"John.1.1": 50, "Heb.11.3": 7},
        "Gen.1.3": {"2Cor.4.6": 12},
    }


def test_qrels_cut_is_inclusive():
    assert build_qrels(XREFS, 3)["Gen.1.1"] == {"John.1.1": 50, "Heb.11.3": 7, "Ps.33.6": 3}


def test_qrels_drop_self_links_and_empty_queries():
    q = build_qrels(XREFS, 1)
    assert "Gen.1.1" not in q["Gen.1.1"]
    assert set(q) == {"Gen.1.1", "Gen.1.2", "Gen.1.3"}
    assert "Gen.1.2" not in build_qrels(XREFS, 5)


# --- split_queries ------------------------------------------------------------------

ELIGIBLE = [f"ot{i}" for i in range(700)] + [f"nt{i}" for i in range(300)]
STRATUM = {q: ("Old Testament" if q.startswith("ot") else "New Testament") for q in ELIGIBLE}


def test_split_sizes_are_exact_and_disjoint():
    dev, test = split_queries(ELIGIBLE, STRATUM, n_dev=101, n_test=55, seed=1)
    assert len(dev) == 101 and len(test) == 55
    assert not set(dev) & set(test)
    assert set(dev) | set(test) <= set(ELIGIBLE)


def test_split_is_stratified():
    dev, test = split_queries(ELIGIBLE, STRATUM, n_dev=200, n_test=100, seed=1)
    assert sum(q.startswith("ot") for q in dev) == 140    # round(200 * 0.7)
    assert sum(q.startswith("ot") for q in test) == 70


def test_split_is_seeded_and_order_invariant():
    a = split_queries(ELIGIBLE, STRATUM, 50, 50, seed=7)
    assert a == split_queries(list(reversed(ELIGIBLE)), STRATUM, 50, 50, seed=7)
    assert a != split_queries(ELIGIBLE, STRATUM, 50, 50, seed=8)
    assert a[0] == sorted(a[0]) and a[1] == sorted(a[1])


def test_split_rejects_oversized_request():
    with pytest.raises(ValueError):
        split_queries(ELIGIBLE, STRATUM, 600, 401, seed=0)


# --- filter_ranking and gains --------------------------------------------------------

def test_filter_drops_query_and_unjudged_keeping_order():
    ranked = ["Isa.2.2", "2Ne.12.2", "Mic.4.1", "Isa.2.3", "2Ne.12.3"]
    judged = {"Isa.2.2", "Mic.4.1", "Isa.2.3"}
    assert filter_ranking(ranked, "Isa.2.2", judged) == ["Mic.4.1", "Isa.2.3"]


def test_gains_binary_and_graded():
    qrel = {"a": 3, "b": 15}
    assert gains(qrel, graded=False) == {"a": 1.0, "b": 1.0}
    assert gains(qrel, graded=True) == pytest.approx({"a": 2.0, "b": 4.0})


# --- evaluate_query -------------------------------------------------------------------

def test_evaluate_query_condenses_before_scoring():
    # Raw: query, BoM, hit, BoM, miss, hit. Condensed: hit, miss, hit.
    ranked = ["Isa.2.2", "2Ne.12.2", "Mic.4.1", "2Ne.12.3", "Gen.1.1", "Isa.2.3"]
    judged = {"Isa.2.2", "Mic.4.1", "Gen.1.1", "Isa.2.3", "Isa.11.9"}
    qrel = {"Mic.4.1": 31, "Isa.2.3": 7, "Isa.11.9": 5}
    m = evaluate_query(ranked, "Isa.2.2", qrel, judged, k=3)

    assert set(m) == set(METRICS)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["recall_capped"] == pytest.approx(2 / 3)
    assert m["mrr"] == 1.0
    assert m["ndcg"] == pytest.approx((1 + 1 / log2(4)) / (1 + 1 / log2(3) + 1 / log2(4)))
    g = {d: log2(1 + v) for d, v in qrel.items()}
    ideal = g["Mic.4.1"] + g["Isa.2.3"] / log2(3) + g["Isa.11.9"] / log2(4)
    assert m["ndcg_graded"] == pytest.approx((g["Mic.4.1"] + g["Isa.2.3"] / log2(4)) / ideal)
    # Raw top 3 without the query: 2Ne.12.2, Mic.4.1, 2Ne.12.3 -> 2 of 3 unjudged.
    assert m["unjudged_share"] == pytest.approx(2 / 3)


# --- paired_bootstrap -------------------------------------------------------------------

def test_bootstrap_identical_models_give_zero():
    a = np.random.default_rng(0).random(100)
    assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0)


def test_bootstrap_constant_shift_is_exact():
    a = np.random.default_rng(0).random(100)
    diff, lo, hi = paired_bootstrap(a + 0.1, a)
    assert diff == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_bootstrap_interval_brackets_difference_and_is_seeded():
    rng = np.random.default_rng(1)
    a, b = rng.random(200), rng.random(200)
    diff, lo, hi = paired_bootstrap(a, b, seed=3)
    assert diff == pytest.approx(np.mean(a - b))
    assert lo < diff < hi
    assert paired_bootstrap(a, b, seed=3) == (diff, lo, hi)


def test_bootstrap_is_paired():
    # Noisy per query but b is always a - 0.05: a paired interval is exact; an
    # unpaired one (resampling a and b separately) would be wide.
    a = np.random.default_rng(2).random(300)
    diff, lo, hi = paired_bootstrap(a, a - 0.05)
    assert hi - lo == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_rejects_bad_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


# --- run and summarize (end to end on a toy corpus) -----------------------------------

def test_run_and_summarize_end_to_end():
    verses = pd.DataFrame({
        "id": ["Gen.1.1", "John.1.1", "Heb.11.3", "Gen.1.3", "2Cor.4.6", "1Ne.1.1"],
        "text": ["beginning", "beginning word", "worlds framed", "light", "light shine", "goodly"],
    })
    judged = set(verses["id"]) - {"1Ne.1.1"}
    qrels = {c: build_qrels(XREFS, c) for c in (5, 10)}

    def score_fn(q):  # word-overlap scorer
        return np.array([len(set(q.split()) & set(t.split())) for t in verses["text"]], float)

    results = run(score_fn, ["Gen.1.1", "Gen.1.3"], verses, qrels, judged, k=2)
    assert len(results) == 4  # both queries have labels at both cuts
    summary = summarize(results)
    assert list(summary.index) == [5, 10]
    assert summary.loc[5, "n_queries"] == 2
    gen11 = results.query("query_id == 'Gen.1.1' and min_votes == 5").iloc[0]
    assert gen11["recall"] == pytest.approx(1.0)   # John.1.1 and Heb.11.3 in condensed top 2
    assert gen11["mrr"] == 1.0


def test_top_m_matches_stable_full_sort_with_ties():
    from rtm.evaluate import _top_m
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.integers(0, 5, 1000).astype(float)  # heavy ties, like BM25's zeros
        for m in (1, 10, 200, 1000, 2000):
            np.testing.assert_array_equal(_top_m(s, m), np.argsort(-s, kind="stable")[:m])
