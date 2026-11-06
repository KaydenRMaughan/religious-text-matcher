# How it works

A working map of the project for whoever is building it: the pipeline, the models and statistics, and the data science decisions still open. Updated every session. The README is the outward-facing summary; [decisions.md](decisions.md) is the dated log of why.

**Who builds what.** Kayden (data science SME) types the retrieval models, metrics and evaluation protocol. Claude builds the data access, packaging, test fixtures and the app, and writes the spec and failing tests for each data science piece before it's typed.

## Pipeline

```
fetch_data.py ──> data/raw/ ──> corpus.py ──> verses (37,706 rows)
  (Gutenberg,       (gitignored,     │            id, title, text, source, book, chapter, verse
   OpenBible)        SHA-256         │
                     manifest)       └──> cross-references (614,179 verse pairs) ──┐
                                                                                   │
query text ──> retriever (BM25 | MiniLM | mpnet) ──> ranked verses ──> metrics <───┘
                                                     (Recall@10, MRR, nDCG@10)
                                       └──> Streamlit app (demo; any corpus via config.local.toml)
```

| Stage | Module | Status | Owner |
|---|---|---|---|
| Fetch | `scripts/fetch_data.py` | Done | Claude |
| Parse and load | `src/rtm/corpus.py` | Done, 14 tests | Claude |
| Tokenizer | `src/rtm/retrieve.py` `tokenize` | Done (KJV verb rule drafted by Claude at Kayden's request) | **Kayden** |
| BM25 baseline | `src/rtm/retrieve.py` `BM25` | Done: index 0.8 s, about 2 ms a query on 37,706 verses | **Kayden** |
| Metrics | `src/rtm/metrics.py` | Done | **Kayden** |
| Evaluation protocol | `src/rtm/evaluate.py` | Spec and failing tests; runner done (`scripts/run_eval.py`) | **Kayden** (Claude: the batch runner) |
| Embedding retrievers | `src/rtm/retrieve.py`, `src/rtm/embed.py` | Spec and failing tests; encoding and cache done | **Kayden** (Claude: model loading and caching) |
| Citation parser (private test) | `src/rtm/citations.py` | Done, 11 tests; resolves 53,060 cited verses in 2,220 talks once the Pearl of Great Price is loaded | Claude |
| Chunker | `src/rtm/chunk.py` | Done, 6 tests | Claude |
| Max-pool and boost | `src/rtm/retrieve.py` | Done (drafted by Claude for review) | **Kayden** |
| Private evaluation | `scripts/run_private_eval.py` | Built; results private | Claude (protocol: Kayden) |
| Chapter task | `scripts/run_chapter_eval.py` | Built | Claude (protocol: Kayden) |
| Search engine | `src/rtm/search.py` | Done, 5 tests | Claude |
| Boost sizing | `scripts/boost_parallels.py` | Done: δ = 0.0006 (Kayden's choice) | Claude (choice: Kayden) |
| App | `app/app.py` | Done: launched and checked 2026-11-06 | Claude |

## Data

- **Verses.** One row per verse. Bible ids are OSIS style (`Gen.1.1`), so they join directly to the labels. Book of Mormon ids use the same pattern (`1Ne.3.7`) but have no labels. Longest verse: 735 characters, about 180 tokens.
- **Labels.** OpenBible.info cross-references: a directed link from one verse to another, with a crowd vote count (median 3, range −86 to 1,312). Ranges are expanded to one row per verse.
- **Label facts that shaped the protocol** (measured 2026-10-30):

  | Votes ≥ | Links | Queries | Median \|R\| | p90 \|R\| |
  |---|---|---|---|---|
  | 3 | 358,554 | 26,723 | 8 | 31 |
  | **5 (primary)** | 186,665 | 19,915 | 4 | 24 |
  | 10 | 71,180 | 7,739 | 5 | 22 |

  Only 0.2% of links go to an adjacent verse and 3.4% stay in the same chapter; 66 are self-links. `and` and `the` each appear in 79% of verses.
- **Private labels.** The conference corpus carries each talk's own `scripture_references` (e.g. `Isa. 55:9`, `D&C 27:15, 17`): a ready-made talk → verse ground truth that covers the Book of Mormon. Private only.

## Models

**BM25** (the baseline every embedding model has to beat). For query *Q* and verse *D*:

score(D, Q) = Σ<sub>q ∈ Q</sub> IDF(q) · f(q, D)(k₁ + 1) / ( f(q, D) + k₁(1 − b + b·|D| / avgdl) )

where *f* is the term count in *D*, |*D*| the verse length, avgdl the mean verse length, and usually k₁ ∈ [1.2, 2.0] and b = 0.75. IDF is Lucene's, ln(1 + (N − n + 0.5)/(n + 0.5)), which stays positive; Okapi's ln((N − n + 0.5)/(n + 0.5)) goes negative for any term in more than half the verses, and rank-bm25 patches that with an ad hoc floor. The tests check scores against rank-bm25's scorer with Lucene's IDF swapped in. k₁ and b are tuned on the dev queries only.

**Tokenizer.** Normalize U+2019 to `'`, lowercase, letter runs with an optional internal apostrophe (`father's` stays one token), no stopwords. Two variants, picked on the dev set: unstemmed, and stemmed (KJV -eth/-est verb endings mapped to modern forms, then Snowball English, which alone leaves `loveth` untouched). An unstemmed-only baseline would be a straw man on KJV text.

**Sentence embeddings.** `all-MiniLM-L6-v2` (384 dimensions, 256-token limit) and `all-mpnet-base-v2` (768 dimensions, 384 tokens). Score = cosine similarity of L2-normalized vectors, which is a dot product, so exact search over 37k verses is one matrix multiply.

**Chunking with max-pooling** (for long documents): split into passages, embed each, and score a document as its best passage, max<sub>i</sub> cos(q, pᵢ).

## Metrics

For a query with relevant set *R* and ranked list *L*:
- **Recall@k** = |R ∩ L<sub>1..k</sub>| / |R|, and capped recall divides by min(|R|, k) instead. Both are reported.
- **MRR@10** = mean over queries of 1 / rank of the first relevant hit (0 if none in the top 10).
- **nDCG@k** = DCG@k / IDCG@k, with DCG@k = Σ<sub>i≤k</sub> gainᵢ / log₂(i + 1). Binary gain is primary; graded gain log₂(1 + votes) is a secondary row.

## Evaluation protocol (settled 2026-10-30)

1. **Relevance threshold:** votes ≥ 5 primary; ≥ 3 and ≥ 10 as robustness rows. What matters is whether the model ranking holds across all three.
2. **Relevance grading:** binary primary, graded nDCG secondary. Votes track popularity as well as relevance.
3. **Queries:** two disjoint seeded samples of 2,000 source verses, stratified by testament. Dev is for tuning (k₁, b, tokenizer variant) and iteration; test is run once per model for the README.
4. **Leakage:** the query verse is removed from its own candidates and self-links are dropped. Neighbors stay in, so the task is "find related verses anywhere", which slightly penalizes lexical models that rank adjacent verses high.
5. **Recall:** capped and uncapped both reported; low uncapped numbers are expected with median |R| of 4–8.
6. **Chunking vs truncation:** tested twice. Privately on the conference corpus (talk → its cited verses; results stay private), with the Doctrine and Covenants added privately so D&C citations resolve. Publicly on a second task over longer units (chapters).
7. **Tokenizer:** as in Models above.
8. **Book of Mormon:** eligible and preferred as answers.
   - *Scoring (8a):* Book of Mormon results are unjudged by OpenBible, so they're dropped from the ranked list before the Bible metrics are computed (condensed lists, Sakai 2007). Each model's Book of Mormon share of the top 10 is reported separately. Book of Mormon accuracy is measured on the private talk-citation labels.
   - *Preference (8b):* a score boost for Book of Mormon verses, sized to win when scores are close: 2 Nephi 12 should outrank Isaiah 2, which it quotes. A source boost never changes the order of Bible verses among themselves, so it leaves the condensed Bible metrics untouched.

## Settled at the app stage (2026-11-06)

- **8b. Boost:** additive δ = 0.0006 on the hybrid's RRF scores, sized on 445 aligned parallel verse pairs. It flips word-for-word twins anywhere in the top 10 and raises the Book of Mormon share of the top 10 from 11.1% to 12.6%. On private dev, any δ > 0 lowered nDCG; the boost is a stated preference. See decisions.md.
- **6a, 6b:** done. The chapter task results are public (README); the private test results are in `results/private/` and stay private.

## Open decisions

- None in the data science. `feat/private-test` merged to `main` 2026-11-06 (`3bf29ed`). Remaining work is shipping: Kayden's review of the Writing post draft (site repo, `writing/posts/religious-text-matcher/`) and the resume bullet, evidence-bank sync, repo hygiene, and flipping the repo public.
- **Budget.** 15–19 hrs; about 13.5 logged as of 2026-11-06.
