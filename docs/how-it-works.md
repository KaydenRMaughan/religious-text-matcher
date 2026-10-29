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
| BM25 baseline | `src/rtm/retrieve.py` | Next | **Kayden** |
| Metrics | `src/rtm/metrics.py` | Next | **Kayden** |
| Evaluation protocol | `src/rtm/evaluate.py` | Next | **Kayden** (Claude: the batch runner) |
| Embedding retrievers | `src/rtm/retrieve.py` | Later | **Kayden** (Claude: model loading and caching) |
| App | `app/app.py` | Later | Claude |

## Data

- **Verses.** One row per verse. Bible ids are OSIS style (`Gen.1.1`), so they join directly to the labels. Book of Mormon ids use the same pattern (`1Ne.3.7`) but have no labels. Longest verse: 735 characters, about 180 tokens.
- **Labels.** OpenBible.info cross-references: a directed link from one verse to another, with a crowd vote count (median 3, range −86 to 1,312). Ranges are expanded to one row per verse.

## Models

**BM25** (the baseline every embedding model has to beat). For query *Q* and verse *D*:

score(D, Q) = Σ<sub>q ∈ Q</sub> IDF(q) · f(q, D)(k₁ + 1) / ( f(q, D) + k₁(1 − b + b·|D| / avgdl) )

where *f* is the term count in *D*, |*D*| the verse length, avgdl the mean verse length, and usually k₁ ∈ [1.2, 2.0] and b = 0.75. `rank-bm25` provides `BM25Okapi`. One option: type BM25 from the formula yourself and test it against `rank-bm25`'s scores.

**Sentence embeddings.** `all-MiniLM-L6-v2` (384 dimensions, 256-token limit) and `all-mpnet-base-v2` (768 dimensions, 384 tokens). Score = cosine similarity of L2-normalized vectors, which is a dot product, so exact search over 37k verses is one matrix multiply.

**Chunking with max-pooling** (for long documents): split into passages, embed each, and score a document as its best passage, max<sub>i</sub> cos(q, pᵢ).

## Metrics

For a query with relevant set *R* and ranked list *L*:
- **Recall@k** = |R ∩ L<sub>1..k</sub>| / |R|. Undefined choice: divide by |R| or by min(|R|, k)?
- **MRR** = mean over queries of 1 / rank of the first relevant hit (0 if none in the cutoff).
- **nDCG@k** = DCG@k / IDCG@k, with DCG@k = Σ<sub>i≤k</sub> gainᵢ / log₂(i + 1). Gain can be binary or graded by votes.

## Open data science decisions (for the next session)

1. **Vote threshold.** 3,512 links have votes ≤ 0. Keep votes ≥ 1? Higher?
2. **Binary or graded relevance** for nDCG. Graded by votes is richer, but votes track popularity as well as relevance.
3. **Query set.** All 29,363 source verses, or a seeded sample (say 2,000, stratified by testament) for faster iteration?
4. **Leakage.** Exclude the query verse itself, obviously. Also its immediate neighbors? Adjacent verses are trivially similar and often cross-referenced, which inflates every model's score.
5. **Recall denominator:** |R| or min(|R|, k) (see Metrics).
6. **Chunking vs truncation can't be tested on verses.** No verse exceeds the token limits, so that comparison needs longer documents. Option: a chapter-level task (retrieve chapters; a chapter is relevant if it contains a cross-reference target). Otherwise the claim moves to the private conference corpus, where it can't be published.
7. **Tokenizer for BM25.** The texts use curly apostrophes (U+2019), so `father’s` and `father's` won't match unless normalized. Also: lowercasing, stopwords (KJV "thee", "thou", "hath"), and whether to stem.
