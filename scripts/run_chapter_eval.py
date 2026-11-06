"""Chapter task: chunking vs truncation on long documents, with public data.

    python scripts/run_chapter_eval.py              # dev and test

A verse is the query; the documents are the 1,189 Bible chapters (median ~650 words,
well past MiniLM's 256-token limit). A chapter is relevant if it contains at least one
>= 5-vote cross-reference target of the query (3 and 10 as robustness rows); the
query's own chapter is excluded from both ranking and labels (decided 2026-11-05).

Models:
- bm25: the chapter's stemmed tokens (k1, b as tuned on verses; not re-tuned).
- <model>_truncated: one embedding per chapter; the model reads only its first 256
  (MiniLM) or 384 (mpnet) tokens, the old app's behavior on long texts.
- <model>_chunked: ~200-word passages; a chapter scores as its best passage (max-pool).
- hybrid: RRF of bm25 and mpnet_chunked.
Nothing is tuned on this task, so test is reported directly; dev is a sanity check.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from _setup import RESULTS, as_float, load_setup

from rtm.chunk import chunk_text
from rtm.embed import embed_corpus
from rtm.evaluate import paired_bootstrap, run, summarize
from rtm.retrieve import BM25, DenseRetriever, rrf_scores, tokenize

K1, B = 0.4, 1.0


def chapters_of(verses: pd.DataFrame) -> pd.DataFrame:
    bible = verses[verses["source"] != "Book of Mormon"].copy()
    bible["chapter_id"] = bible["id"].str.rsplit(".", n=1).str[0]
    return (bible.groupby("chapter_id", sort=False)["text"].apply(" ".join)
            .reset_index().rename(columns={"chapter_id": "id"}))


def chapter_qrels(qrels, chapter_of):
    out = {}
    for cut, labels in qrels.items():
        out[cut] = {}
        for q, rel in labels.items():
            chs = {chapter_of[v] for v in rel if v in chapter_of} - {chapter_of.get(q)}
            if chs:
                out[cut][q] = {c: 1 for c in chs}
    return out


def main() -> None:
    s = load_setup()
    chapters = chapters_of(s.verses)
    chapter_of = {v: v.rsplit(".", 1)[0] for v in s.verses["id"] if v in s.judged}
    qrels = chapter_qrels(s.qrels, chapter_of)
    exclude = {q: {chapter_of[q]} for q in s.dev + s.test}
    words = chapters["text"].str.split().map(len)
    print(f"{len(chapters):,} chapters; words per chapter median {words.median():.0f}, "
          f"p90 {words.quantile(.9):.0f}")

    verse_row = {t: i for i, t in enumerate(s.verses["text"])}
    bm25 = BM25([tokenize(t, stem=True) for t in chapters["text"]], k1=K1, b=B)
    models = {"bm25": lambda q: bm25.scores(tokenize(q, stem=True))}
    chunks = [chunk_text(t) for t in chapters["text"]]
    starts = np.cumsum([0] + [len(c) for c in chunks])[:-1]
    for key in ("minilm", "mpnet"):
        verse_vecs = embed_corpus(s.verses["text"].tolist(), key)
        whole = DenseRetriever(embed_corpus(chapters["text"].tolist(), key))
        passages = DenseRetriever(embed_corpus([c for cs in chunks for c in cs], key))
        models[f"{key}_truncated"] = (lambda r, v: lambda q: r.scores(v[verse_row[q]]))(whole, verse_vecs)
        models[f"{key}_chunked"] = (
            lambda r, v: lambda q: np.maximum.reduceat(r.scores(v[verse_row[q]]), starts)
        )(passages, verse_vecs)
    models["hybrid"] = lambda q: rrf_scores([models["bm25"](q), models["mpnet_chunked"](q)])

    judged = set(chapters["id"])
    for split, queries in (("dev", s.dev), ("test", s.test)):
        results = {}
        for name, fn in models.items():
            start = time.perf_counter()
            results[name] = run(fn, queries, chapters, qrels, judged,
                                query_text=s.verses.set_index("id")["text"], exclude=exclude)
            results[name].to_csv(RESULTS / f"chapter_{split}_{name}.csv", index=False)
            print(f"\n{name} on chapter {split} ({time.perf_counter() - start:.0f} s)")
            print(summarize(results[name]).drop(columns=["unjudged_share"]).round(3).to_string())
        rows = []
        for a, b in [("minilm_chunked", "minilm_truncated"), ("mpnet_chunked", "mpnet_truncated"),
                     ("mpnet_chunked", "bm25"), ("hybrid", "bm25")]:
            both = results[a].merge(results[b], on=["query_id", "min_votes"], suffixes=("", "_b"))
            for cut, g in both.groupby("min_votes"):
                d, lo, hi = paired_bootstrap(g["ndcg"], g["ndcg_b"])
                rows.append({"comparison": f"{a} - {b}", "min_votes": cut,
                             "ndcg diff [95% CI]": f"{as_float(d):+.3f} [{as_float(lo):+.3f}, {as_float(hi):+.3f}]"})
        print(f"\nPaired bootstrap, chapter {split} (nDCG@10):")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
