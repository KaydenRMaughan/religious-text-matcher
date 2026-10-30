from math import log2

import pytest

from rtm.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

RANKED = ["a", "x", "b", "y"]
RELEVANT = {"a", "b", "c"}


# --- recall ---------------------------------------------------------------------

def test_recall_uncapped_divides_by_relevant_set_size():
    assert recall_at_k(RANKED, RELEVANT, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, k=4) == pytest.approx(2 / 3)


def test_recall_capped_divides_by_min_of_set_size_and_k():
    assert recall_at_k(RANKED, RELEVANT, k=2, capped=True) == pytest.approx(1 / 2)
    assert recall_at_k(RANKED, RELEVANT, k=4, capped=True) == pytest.approx(2 / 3)


def test_perfect_top_k_with_large_relevant_set():
    relevant = {f"r{i}" for i in range(30)}
    ranked = [f"r{i}" for i in range(10)]
    assert recall_at_k(ranked, relevant, k=10, capped=True) == 1.0
    assert recall_at_k(ranked, relevant, k=10) == pytest.approx(1 / 3)


def test_ranked_list_shorter_than_k():
    assert recall_at_k(["a"], {"a", "b"}, k=10) == pytest.approx(1 / 2)
    assert recall_at_k(["a"], {"a", "b"}, k=10, capped=True) == pytest.approx(1 / 2)


def test_recall_no_hits_is_zero():
    assert recall_at_k(["x", "y"], RELEVANT, k=2) == 0.0


# --- reciprocal rank ----------------------------------------------------------------

def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["x", "y", "a", "b"], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], RELEVANT) == 1.0


def test_reciprocal_rank_respects_cutoff():
    assert reciprocal_rank(["x", "y", "a"], RELEVANT, k=2) == 0.0
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


# --- nDCG ---------------------------------------------------------------------------

def test_ndcg_binary_hand_computed():
    dcg = 1 / log2(3) + 1 / log2(5)  # hits at ranks 2 and 4
    idcg = 1 / log2(2) + 1 / log2(3)
    assert ndcg_at_k(RANKED, {"a": 1.0, "b": 1.0}, k=4) == pytest.approx(
        (1 / log2(2) + 1 / log2(4)) / idcg
    )
    assert ndcg_at_k(["x", "a", "y", "b"], {"a": 1.0, "b": 1.0}, k=4) == pytest.approx(dcg / idcg)


def test_ndcg_graded_rewards_putting_stronger_link_first():
    gains = {"a": 3.0, "b": 1.0}
    idcg = 3.0 + 1.0 / log2(3)
    assert ndcg_at_k(["b", "a"], gains, k=2) == pytest.approx((1.0 + 3.0 / log2(3)) / idcg)
    assert ndcg_at_k(["a", "b"], gains, k=2) == pytest.approx(1.0)


def test_ndcg_ideal_list_is_truncated_at_k():
    # Three relevant ids but k=2: the ideal list holds only two, so two hits score 1.0.
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0, "c": 1.0}, k=2) == pytest.approx(1.0)


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a": 1.0}, k=2) == 0.0


# --- shared validation -----------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda: recall_at_k(RANKED, set(), k=2),
        lambda: recall_at_k(RANKED, RELEVANT, k=0),
        lambda: recall_at_k(["a", "a"], RELEVANT, k=2),
        lambda: reciprocal_rank(RANKED, set()),
        lambda: reciprocal_rank(["a", "a"], RELEVANT),
        lambda: ndcg_at_k(RANKED, {}, k=2),
        lambda: ndcg_at_k(RANKED, {"a": 0.0}, k=2),
        lambda: ndcg_at_k(["a", "a"], {"a": 1.0}, k=2),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
