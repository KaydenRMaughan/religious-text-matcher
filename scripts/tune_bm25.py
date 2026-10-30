"""Tune stemmed BM25's k1 and b on the dev queries (votes >= 5, nDCG@10).

    python scripts/tune_bm25.py

Writes results/tune_bm25.csv and prints the best combinations. Dev only, by design.
"""
from __future__ import annotations

import time

from _setup import RESULTS, bm25_score_fn, load_setup

from rtm.evaluate import grid_search
from rtm.retrieve import tokenize

# Widened after a smoke run put the best value on the low-k1, high-b edge.
GRID = {
    "k1": [0.2, 0.4, 0.6, 0.9, 1.2, 1.5, 1.8],
    "b": [0.3, 0.5, 0.7, 0.85, 1.0],
}


def main() -> None:
    s = load_setup()
    tokens = [tokenize(t, stem=True) for t in s.verses["text"]]  # tokenize once, reuse
    start = time.perf_counter()
    out = grid_search(
        lambda k1, b: bm25_score_fn(s.verses, k1, b, tokens),
        GRID, s.dev, s.verses, s.qrels, s.judged,
    )
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "tune_bm25.csv", index=False)
    print(f"{len(out)} combinations in {time.perf_counter() - start:.0f} s; best first:")
    print(out.head(10).round(4).to_string(index=False))
    print("\nAs a k1 x b table (nDCG@10):")
    print(out.pivot(index="k1", columns="b", values="ndcg").round(4).to_string())


if __name__ == "__main__":
    main()
