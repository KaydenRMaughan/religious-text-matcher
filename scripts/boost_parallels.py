"""Size the Book of Mormon boost on the known parallels (decision 8b; public data only).

    python scripts/boost_parallels.py

For each delta (in hybrid RRF units) it reports:
- how often a Book of Mormon verse outranks the Bible verse it quotes, on queries
  OpenBible links (>= 5 votes) to that Bible verse;
- the Book of Mormon share of the top 10 on the 2,000 public dev queries (flooding check).

Pairs come from the parallel chapters below, aligned verse by verse on stemmed-token
Jaccard (>= 0.5), since the two books' verse numbering doesn't always match. This is a
diagnostic for Kayden's choice of delta, not a reported metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _setup import RESULTS, hybrid_score_fn, load_setup

from rtm.retrieve import tokenize

# (Book of Mormon book, chapters) <-> (Bible book, chapters), chapter for chapter
PARALLELS = [
    ("2 Nephi", range(12, 25), "Isaiah", range(2, 15)),
    ("1 Nephi", range(20, 22), "Isaiah", range(48, 50)),
    ("Mosiah", [14], "Isaiah", [53]),
    ("3 Nephi", [22], "Isaiah", [54]),
    ("3 Nephi", range(12, 15), "Matthew", range(5, 8)),
]
MIN_JACCARD = 0.5
DELTAS = [0.0, 0.00002, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002]


def align_pairs(verses: pd.DataFrame) -> pd.DataFrame:
    toks = {i: set(tokenize(t, stem=True)) for i, t in zip(verses["id"], verses["text"])}
    rows = []
    for bom_book, bom_chs, bib_book, bib_chs in PARALLELS:
        for bc, ac in zip(bom_chs, bib_chs):
            bom = verses[(verses.book == bom_book) & (verses.chapter == bc)]
            bib = verses[(verses.book == bib_book) & (verses.chapter == ac)]
            for b in bom["id"]:
                sims = [(len(toks[b] & toks[a]) / len(toks[b] | toks[a]), a) for a in bib["id"]]
                j, a = max(sims)
                if j >= MIN_JACCARD:
                    rows.append({"bom": b, "bible": a, "jaccard": j})
    return pd.DataFrame(rows)


def main() -> None:
    s = load_setup()
    v = s.verses
    pos = {i: n for n, i in enumerate(v["id"])}
    is_bom = (v["source"] == "Book of Mormon").to_numpy()
    text = v.set_index("id")["text"]
    score = hybrid_score_fn(v, 0.4, 1.0)

    pairs = align_pairs(v)
    parallel_bible = set(pairs["bible"])
    excluded = set(v.loc[v.set_index(["book", "chapter"]).index.isin(
        [(bk, c) for _, _, bk, chs in PARALLELS for c in chs]), "id"])
    print(f"{len(pairs)} aligned pairs (median Jaccard {pairs.jaccard.median():.2f}) "
          f"over {len(parallel_bible)} Bible verses")

    # (query, pair) cases: query links to the pair's Bible verse at >= 5 votes
    cases = []
    for q, rel in s.qrels[5].items():
        if q in excluded:
            continue
        for a in parallel_bible & rel.keys():
            for b in pairs.loc[pairs.bible == a, "bom"]:
                cases.append((q, a, b))
    print(f"{len(cases)} query-pair cases from {len({c[0] for c in cases})} queries")

    by_query: dict[str, np.ndarray] = {}
    gaps, a_ranks, b_ranks = [], [], []
    for q, a, b in cases:
        if q not in by_query:
            by_query[q] = score(text[q])
        sc = by_query[q].copy()
        sc[pos[q]] = -np.inf                      # the query verse isn't a candidate
        order = np.argsort(-sc, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        gaps.append(sc[pos[a]] - sc[pos[b]])      # delta needed for b to pass a
        a_ranks.append(rank[pos[a]])
        b_ranks.append(rank[pos[b]])
    gaps, a_ranks, b_ranks = map(np.asarray, (gaps, a_ranks, b_ranks))
    print(f"at delta 0: Bible side median rank {np.median(a_ranks):.0f}, "
          f"Book of Mormon side {np.median(b_ranks):.0f}; "
          f"Book of Mormon already ahead in {np.mean(gaps < 0):.0%}")

    dev_scores = []
    for q in s.dev:
        sc = score(text[q])
        sc[pos[q]] = -np.inf
        dev_scores.append(sc)

    rows = []
    for d in DELTAS:
        shares = [is_bom[np.argsort(-(sc + d * is_bom), kind="stable")[:10]].mean()
                  for sc in dev_scores]
        rows.append({"delta": d,
                     "bom_ahead": np.mean(gaps < d),
                     "bom_ahead_when_bible_top10": np.mean((gaps < d)[a_ranks <= 10]),
                     "dev_bom_share_top10": np.mean(shares)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "boost_parallels.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
