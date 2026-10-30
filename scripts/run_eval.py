"""Run the retrieval evaluation, write per-query results, and compare models to BM25.

    python scripts/run_eval.py                          # all models on dev
    python scripts/run_eval.py --models bm25 minilm     # a subset
    python scripts/run_eval.py --k1 1.2 --b 0.6         # BM25 with tuned parameters
    python scripts/run_eval.py --split test             # the reported run: once, at the end

The query samples are seeded, so every run and every model sees the same queries.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from _setup import RESULTS, as_float, bm25_score_fn, dense_score_fn, load_setup

from rtm.evaluate import paired_bootstrap, run, summarize

COMPARE = ("ndcg", "recall_capped", "mrr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--models", nargs="+", default=["bm25", "minilm", "mpnet"],
                        choices=["bm25", "minilm", "mpnet"])
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    s = load_setup()
    queries = s.dev if args.split == "dev" else s.test
    RESULTS.mkdir(exist_ok=True)
    results = {}
    for model in args.models:
        score_fn = (bm25_score_fn(s.verses, args.k1, args.b) if model == "bm25"
                    else dense_score_fn(s.verses, model))
        start = time.perf_counter()
        results[model] = run(score_fn, queries, s.verses, s.qrels, s.judged)
        results[model].to_csv(RESULTS / f"{args.split}_{model}.csv", index=False)
        label = f"bm25 (stemmed, k1={args.k1}, b={args.b})" if model == "bm25" else model
        print(f"\n{label} on {args.split} ({time.perf_counter() - start:.0f} s)")
        print(summarize(results[model]).round(3).to_string())

    if "bm25" in results and len(results) > 1:
        print("\nPaired bootstrap vs BM25: mean difference [95% interval]")
        rows = []
        for model in (m for m in results if m != "bm25"):
            both = results[model].merge(results["bm25"], on=["query_id", "min_votes"],
                                        suffixes=("", "_bm25"))
            for cut, g in both.groupby("min_votes"):
                row = {"model": model, "min_votes": cut}
                for metric in COMPARE:
                    d, lo, hi = paired_bootstrap(g[metric], g[f"{metric}_bm25"])
                    row[metric] = f"{as_float(d):+.3f} [{as_float(lo):+.3f}, {as_float(hi):+.3f}]"
                rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS / f"{args.split}_compare.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
