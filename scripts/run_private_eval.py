"""Private evaluation: each conference talk is a query; its footnoted verses are the labels.

    python scripts/run_private_eval.py                 # all models on private dev and test
    python scripts/run_private_eval.py --tune-boost    # also tune the Book of Mormon boost

Results are private: written to results/private/ (gitignored), never published. Only
aggregate findings, with no talk text, may appear in public docs.

Protocol (decided 2026-11-05):
- Pool: KJV, Book of Mormon, D&C and Pearl of Great Price verses (41,995), all judged.
- Labels: verse-level citations (chapter-only citations dropped), binary.
- Talks: 2,220 with at least one cited verse, split 50/50 into dev and test, seeded and
  stratified by decade. The boost is tuned on dev; everything is reported on test.
- Long queries: chunked into ~200-word sentence windows (1 sentence overlap), scored
  by each verse's best chunk (max-pool), compared against truncation (the old app).
- Label sets reported as "cuts": 1 = all cited verses, 2 = Book of Mormon citations
  only (how well each model finds the Book of Mormon verses speakers chose).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from _setup import as_float

from rtm.chunk import chunk_text
from rtm.citations import parse_all
from rtm.corpus import REPO_ROOT, load_corpus, load_local_config, load_scripture, load_scriptures_json
from rtm.embed import embed_corpus, encode
from rtm.evaluate import grid_search, paired_bootstrap, run, split_queries, summarize
from rtm.retrieve import (
    BM25, DenseRetriever, boost, max_pool_scores, rrf_scores, tokenize,
)

SEED = 20261105
K1, B = 0.4, 1.0                       # tuned on public dev (verse queries)
CACHE = REPO_ROOT / "data" / "private" / "embeddings"
OUT = REPO_ROOT / "results" / "private"
LABEL_SETS = {1: "all citations", 2: "Book of Mormon citations"}
BOOST_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008]


def load():
    config = load_local_config()
    p = config["private"]
    path = lambda x: REPO_ROOT / x
    pool = pd.concat([
        load_scripture(),
        load_scriptures_json(path(p["dc_json"]), "Doctrine and Covenants", "DC"),
        load_scriptures_json(path(p["pgp_json"]), "Pearl of Great Price"),
    ], ignore_index=True)
    talks = load_corpus(config["corpus"]["path"])
    order = pool["id"].tolist()
    bom = set(pool.loc[pool["source"] == "Book of Mormon", "id"])
    labels = {t.id: parse_all(t.citations, order).verses for t in talks.itertuples()}
    qrels = {
        1: {t: {v: 1 for v in vs} for t, vs in labels.items() if vs},
        2: {t: {v: 1 for v in vs if v in bom} for t, vs in labels.items() if set(vs) & bom},
    }
    decade = {t.id: f"{t.year // 10 * 10}s" for t in talks.itertuples()}
    eligible = sorted(qrels[1])
    half = len(eligible) // 2
    dev, test = split_queries(eligible, decade, half, len(eligible) - half, SEED)
    return pool, talks, qrels, dev, test


def pool_vectors(pool: pd.DataFrame, key: str) -> np.ndarray:
    """Public verses reuse the public cache; D&C and Pearl of Great Price go to the private one."""
    public = pool["source"].isin(["Old Testament", "New Testament", "Book of Mormon"])
    return np.vstack([
        embed_corpus(pool.loc[public, "text"].tolist(), key),
        embed_corpus(pool.loc[~public, "text"].tolist(), key, cache_dir=CACHE),
    ])


def talk_vectors(talks: pd.DataFrame, key: str):
    """Truncated whole-talk vectors, and chunk vectors with each talk's row range."""
    whole = embed_corpus(talks["text"].tolist(), key, cache_dir=CACHE)
    chunks = [chunk_text(t) for t in talks["text"]]
    flat = embed_corpus([c for cs in chunks for c in cs], key, cache_dir=CACHE)
    bounds = np.cumsum([0] + [len(cs) for cs in chunks])
    return whole, flat, bounds


def memo(fn):
    """Cache a score function's output per query text (float32; cleared per split)."""
    cache = {}

    def wrapped(q):
        if q not in cache:
            cache[q] = np.asarray(fn(q), dtype=np.float32)
        return cache[q]
    wrapped.clear = cache.clear
    return wrapped


def build_models(pool, talks):
    by_text = {t: i for i, t in enumerate(talks["text"])}
    bm25 = BM25([tokenize(t, stem=True) for t in pool["text"]], k1=K1, b=B)
    models = {"bm25": memo(lambda q: bm25.scores_counted(tokenize(q, stem=True)))}
    for key in ("minilm", "mpnet"):
        retriever = DenseRetriever(pool_vectors(pool, key))
        whole, flat, bounds = talk_vectors(talks, key)
        models[f"{key}_truncated"] = (lambda r, w: lambda q: r.scores(w[by_text[q]]))(retriever, whole)
        models[f"{key}_chunked"] = memo((
            lambda r, f, b: lambda q: max_pool_scores(r, f[b[by_text[q]]:b[by_text[q] + 1]])
        )(retriever, flat, bounds))
    models["hybrid"] = lambda q: rrf_scores([models["bm25"](q), models["mpnet_chunked"](q)])
    return models


def bom_share(score_fn, talk_text, query_ids, is_bom, k=10):
    """Mean share of the raw top k that is Book of Mormon (diagnostic)."""
    tops = [np.argsort(-np.asarray(score_fn(talk_text[q])), kind="stable")[:k] for q in query_ids]
    return float(np.mean([is_bom[t].mean() for t in tops]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune-boost", action="store_true")
    args = parser.parse_args()

    pool, talks, qrels, dev, test = load()
    print(f"pool {len(pool):,} verses; {len(qrels[1]):,} talks with cited verses "
          f"(dev {len(dev)}, test {len(test)}); {len(qrels[2]):,} cite the Book of Mormon")
    judged = set(pool["id"])
    is_bom = (pool["source"] == "Book of Mormon").to_numpy()
    talk_text = talks.set_index("id")["text"]
    models = build_models(pool, talks)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.tune_boost:
        start = time.perf_counter()
        grid = grid_search(lambda delta: lambda q: boost(models["hybrid"](q), is_bom, delta),
                           {"delta": BOOST_GRID}, dev, pool, qrels, judged, cut=1,
                           query_text=talk_text)
        grid.to_csv(OUT / "tune_boost.csv", index=False)
        print(f"\nBoost on hybrid, private dev ({time.perf_counter() - start:.0f} s):")
        print(grid.round(4).to_string(index=False))
        delta = float(grid.loc[0, "delta"])
        models["hybrid_boost"] = lambda q: boost(models["hybrid"](q), is_bom, delta)
        print(f"chosen delta = {delta}")

    for split, queries in (("dev", dev), ("test", test)):
        if split == "test":                           # keep memory to one split
            for fn in models.values():
                getattr(fn, "clear", lambda: None)()
        results = {}
        for name, fn in models.items():
            start = time.perf_counter()
            results[name] = run(fn, queries, pool, qrels, judged, query_text=talk_text)
            results[name].to_csv(OUT / f"{split}_{name}.csv", index=False)
            summary = summarize(results[name]).drop(columns=["unjudged_share"])
            summary.index = [LABEL_SETS[c] for c in summary.index]
            share = bom_share(fn, talk_text, queries, is_bom)
            print(f"\n{name} on private {split} ({time.perf_counter() - start:.0f} s); "
                  f"Book of Mormon share of top 10: {share:.3f}")
            print(summary.round(3).to_string())
        rows = []
        for a, b in [("mpnet_chunked", "mpnet_truncated"), ("minilm_chunked", "minilm_truncated"),
                     ("hybrid", "bm25"), ("mpnet_chunked", "bm25"), ("hybrid_boost", "hybrid")]:
            if a not in results:
                continue
            both = results[a].merge(results[b], on=["query_id", "min_votes"], suffixes=("", "_b"))
            for cut, g in both.groupby("min_votes"):
                d, lo, hi = paired_bootstrap(g["ndcg"], g["ndcg_b"])
                rows.append({"comparison": f"{a} - {b}", "labels": LABEL_SETS[cut],
                             "ndcg diff [95% CI]": f"{as_float(d):+.3f} [{as_float(lo):+.3f}, {as_float(hi):+.3f}]"})
        print(f"\nPaired bootstrap, private {split} (nDCG@10):")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
