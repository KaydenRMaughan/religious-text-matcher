"""Run the retrieval evaluation and write per-query results to results/.

    python scripts/run_eval.py                  # BM25, plain and stemmed, on dev
    python scripts/run_eval.py --split test     # the reported run: once per final model

The query samples are seeded (SEED), so every run and every model sees the same queries.
"""
from __future__ import annotations

import argparse
import time

from rtm.corpus import RAW_DIR, REPO_ROOT, load_cross_references, load_scripture
from rtm.evaluate import PRIMARY_CUT, VOTE_CUTS, build_qrels, run, split_queries, summarize
from rtm.retrieve import BM25, tokenize

SEED = 20261030
N_DEV = N_TEST = 2000
RESULTS = REPO_ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    verses = load_scripture()
    xrefs = load_cross_references(RAW_DIR / "cross_references.txt", verses["id"].tolist())
    judged = set(verses.loc[verses["source"] != "Book of Mormon", "id"])
    qrels = {cut: build_qrels(xrefs, cut) for cut in VOTE_CUTS}
    stratum = verses.set_index("id")["source"].to_dict()
    dev, test = split_queries(qrels[PRIMARY_CUT].keys(), stratum, N_DEV, N_TEST, SEED)
    queries = dev if args.split == "dev" else test

    RESULTS.mkdir(exist_ok=True)
    for stem in (False, True):
        name = f"bm25_{'stemmed' if stem else 'plain'}"
        bm25 = BM25([tokenize(t, stem) for t in verses["text"]])
        start = time.perf_counter()
        results = run(lambda q: bm25.scores(tokenize(q, stem)), queries, verses, qrels, judged)
        results.to_csv(RESULTS / f"{args.split}_{name}.csv", index=False)
        print(f"\n{name} on {args.split} ({time.perf_counter() - start:.0f} s)")
        print(summarize(results).round(3).to_string())


if __name__ == "__main__":
    main()
