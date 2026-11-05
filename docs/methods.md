# Methods

How the matcher and its evaluation work: the code, the statistics, and why each choice
was made. Written at a working practitioner's level. [how-it-works.md](how-it-works.md)
is the map of modules and status; [decisions.md](decisions.md) is the dated log of
choices. This file explains the reasoning.

## 1. The problem

Given a passage (a verse, a paragraph of a talk, a typed question), return the passages
in a corpus most related to it. That's **retrieval**: score every document against the
query and rank. The project compares two families of retriever:

- **Lexical (BM25):** related means sharing distinctive words.
- **Semantic (sentence embeddings):** related means close in meaning, even with
  different words.

To compare them honestly, it needs labels it didn't make up: OpenBible.info's
crowd-voted cross-references, about 600k verse-to-verse links.

## 2. Pipeline

```
data/raw/ (Gutenberg KJV + Book of Mormon, OpenBible)       scripts/fetch_data.py
   │
   ├─ verses (37,706)          ─┐                            src/rtm/corpus.py
   └─ cross-references (614k)  ─┤
                                │
  tokenize ─> BM25 index        │                            src/rtm/retrieve.py
  encode   ─> embedding matrix  │   (cached .npy)            src/rtm/embed.py
                                ▼
  build_qrels ─> split_queries ─> run: score, rank, condense, score metrics
                                                             src/rtm/evaluate.py
                                                             src/rtm/metrics.py
  summarize + paired_bootstrap ─> results/                   scripts/run_eval.py
```

## 3. Text to tokens (`tokenize`)

BM25 only sees tokens, so the tokenizer decides what "the same word" means.

1. Curly apostrophes (’) become straight ones ('). The texts use only the curly form;
   people type the straight one.
2. Lowercase.
3. A token is a run of letters, with at most one internal apostrophe: `father's` stays
   one token, and `god-ward` splits into `god`, `ward`.
4. No stopword list. Common words get near-zero weight from IDF anyway (Section 4).
5. Stemmed variant (the chosen baseline): the KJV verb endings *-eth* and *-est* are
   rewritten to *-es* (`loveth` becomes `loves`), then the Snowball stemmer reduces
   inflections (`loves`, `loved`, `loving` all become `love`). A list of 57 words that
   only look like verb forms (`priest`, `harvest`, `teeth`, `Nazareth`) is left alone.
   The list came from a corpus scan: words of 5+ letters ending in *-eth*/*-est* with no
   modern form anywhere in the corpus.

Stemming shrinks the vocabulary from 14,109 to 9,090 terms. On dev it was a small but
real gain (nDCG@10 +0.006), with the usual pattern: a few big wins, many small losses.

## 4. BM25

For query *Q* and verse *D*:

score(D, Q) = Σ<sub>q ∈ Q</sub> IDF(q) · f(q,D)·(k₁+1) / ( f(q,D) + k₁·(1 − b + b·|D|/avgdl) )

Three ideas multiplied together:

- **Rarity, IDF(q) = ln(1 + (N − n + 0.5)/(n + 0.5)).** n is the number of verses
  containing q. A word in 3 verses scores about 9; a word in most verses scores near 0.
  This is Lucene's form. The classic Okapi form drops the "1 +" and goes negative for
  words in more than half the verses; `and` is in 79% of verses.
- **Term-frequency saturation, k₁.** The fraction rises with f, the count of q in D, but
  levels off: the first occurrence matters most. k₁ sets how quickly it levels off.
  At k₁ = 0 only presence counts; as k₁ grows, repeats keep counting.
- **Length normalization, b.** |D|/avgdl compares the verse's length to the average. With
  b = 1, a word in a long verse counts for less (it's more likely there by chance); with
  b = 0, length is ignored.

**Implementation.** An inverted index maps each term to the verses containing it and
the counts: `"shepherd" → (verse indexes, counts)`. A query touches only those verses,
one NumPy expression per query term, so scoring all 37,706 verses takes about 1 ms. The
length part of the denominator is precomputed once.

**Tuning on dev** (35 combinations, nDCG@10 at votes ≥ 5): the best was **k₁ = 0.4,
b = 1.0**, against 1.5 and 0.75 by default. That's +0.005 nDCG (interval +0.002 to
+0.008). Both values make sense for verses:
- Low k₁: in a 25-word verse, a word appearing twice says little more than once.
- b = 1: long verses (genealogies, lists) match many queries by accident, so full
  length normalization helps.

The surface is flat near the top: the best three cells are within 0.001.

## 5. Sentence embeddings

A sentence-embedding model is a transformer that reads a text and outputs one vector:
it embeds each token in context, then averages them. Training pushes texts with similar
meaning toward similar directions. Two models:

| Model | Dimensions | Token limit | Corpus encode (CPU) | Trained on |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 256 | 2 min | ~1B modern sentence pairs |
| all-mpnet-base-v2 | 768 | 384 | 15 min | the same |

**Cosine similarity.** Meaning is carried by direction, so the score is the cosine of the
angle between query and verse vectors:

cos(q, d) = q·d / (‖q‖ ‖d‖)

`l2_normalize` scales every vector to length 1. Then cosine is a plain dot product, and
scoring every verse is one matrix-vector product (`DenseRetriever.scores`). A zero vector
has no direction and is left as zeros rather than divided by zero.

**Exact search.** 37,706 × 768 is small. Approximate indexes such as FAISS trade a little
accuracy for speed at millions of documents; here they'd add a dependency for nothing.

**Truncation and chunking.** Each model reads at most 256 or 384 tokens and ignores the
rest. No verse is that long (the longest is about 180 tokens), but conference talks are
(median about 1,700 words). For long documents the fix is chunking: split into passages,
embed each one, and score the document by its best passage (max-pooling). It's tested
where it matters: the chapter task and the private conference test.

## 6. Labels and protocol (`evaluate.py`)

**Qrels.** `build_qrels` turns the cross-references into "query → {relevant verse →
votes}". The primary cut is votes ≥ 5 (19,915 queries, a median of 4 relevant verses
each), with ≥ 3 and ≥ 10 as robustness checks. A conclusion should hold at all three.
Self-links are dropped.

**Dev and test.** `split_queries` draws two disjoint samples of 2,000 queries, seeded and
stratified by testament (67% Old, 33% New, as in the pool). Everything tunable (k₁, b,
the tokenizer, the hybrid) is chosen on dev. Test is scored once per final model and is
what the README reports. Choosing the best of 35 BM25 settings on dev inflates dev's
number slightly (a selection effect); test is untouched, so it doesn't.

**Condensed lists (Book of Mormon handling).** The labels cover only the Bible, so a
Book of Mormon verse in a ranking is *unjudged*, not wrong. 2 Nephi 12:2 quotes Isaiah
2:2 almost word for word, and scoring it as a miss would penalize the right answer.
`filter_ranking` removes unjudged verses (and the query itself) before scoring, which is
the standard approach for incomplete judgments (Sakai, 2007). Book of Mormon results are
still eligible and still shown in the app. `unjudged_share` reports how much of each
model's raw top 10 they take, about 11–14%.

**Neighbors stay in.** Only 0.2% of links go to an adjacent verse, so neighbors don't
inflate scores. They cost a little, because lexical models rank them high.

## 7. Metrics (`metrics.py`)

Per query, on the condensed top 10. *R* is the relevant set; the table shows the mean
over queries.

| Metric | Formula | Rewards |
|---|---|---|
| Recall@10 | \|R ∩ top 10\| / \|R\| | finding many of the cross-references |
| Capped recall@10 | \|R ∩ top 10\| / min(\|R\|, 10) | the same, with a perfect top 10 scoring 1.0 even when \|R\| > 10 |
| MRR@10 | 1 / rank of the first hit (0 if none) | putting one good answer first |
| nDCG@10 | Σ gainᵢ / log₂(i + 1), divided by the best possible | many good answers, near the top |

nDCG is the primary metric, because it uses the whole top 10 and weights position.
Binary gain (1 per relevant verse) is primary. Graded gain, log₂(1 + votes), is
secondary: votes measure popularity as well as relevance.

**Worked example.** Relevant {a, b}, ranking [x, a, y, b]:
- DCG = 1/log₂3 + 1/log₂5 = 0.631 + 0.431 = 1.062
- ideal DCG = 1 + 0.631 = 1.631
- nDCG = 0.65

## 8. Uncertainty: the paired bootstrap

Two models scored on the same 2,000 queries give 2,000 paired differences, dᵢ. Is
their mean, d̄, real or query-sampling noise? `paired_bootstrap` resamples query indexes
with replacement 2,000 times, recomputes d̄ each time, and reports the 2.5th and 97.5th
percentiles.

- **Why paired:** queries vary enormously in difficulty, and both models face the same
  hard ones. Differencing first removes that shared variation, so the interval reflects
  only how much the models disagree. An unpaired comparison would be far wider.
- **How to read it:** an interval excluding 0 means the gap survives resampling of
  queries. Whether it's large enough to matter is a separate judgment.
- **What it doesn't cover:** uncertainty in the labels themselves, or in which corpus
  was chosen.

## 9. Results on dev (votes ≥ 5, BM25 tuned)

| Model | Capped recall@10 | MRR@10 | nDCG@10 | Δ nDCG vs BM25 [95% CI] |
|---|---|---|---|---|
| BM25, stemmed (k₁ 0.4, b 1.0) | 0.231 | 0.359 | 0.216 | — |
| MiniLM | 0.172 | 0.291 | 0.163 | −0.053 [−0.063, −0.044] |
| mpnet | 0.184 | 0.298 | 0.171 | −0.045 [−0.055, −0.034] |
| BM25 + mpnet, RRF (exploratory) | 0.249 | 0.364 | 0.224 | +0.008 [+0.001, +0.015] |

**Why the embeddings lose.** Two reasons, both measurable:
1. **The labels are lexical.** The cross-references descend mostly from the *Treasury of
   Scripture Knowledge* (1830s), assembled much like a concordance. 47% of labeled
   links share a rare stemmed word, against 4% of random verse pairs. BM25 is built to
   find exactly that.
2. **Domain shift.** Both models were trained on modern English. KJV syntax and
   vocabulary (*thee*, *hath*, *shew*) are far from their training data.

**Where each model helps.** Splitting the dev links by whether the two verses share a
rare word:

| Recall@10 on links that… | BM25 | MiniLM | mpnet |
|---|---|---|---|
| share a rare word (8,810) | 0.205 | 0.156 | 0.168 |
| share none (9,881) | 0.006 | 0.010 | 0.014 |

mpnet finds 2.3x as many of the non-lexical links as BM25, but no model finds many.
Those links are mostly thematic or typological (a prophecy linked to its fulfillment)
and need knowledge, not sentence similarity.

**Hybrid.** Reciprocal rank fusion (RRF) adds 1/(60 + rank) from each model's ranking. It
needs no score calibration between models. It beat BM25 at every vote cut, with every
interval excluding 0 (+0.014, +0.008, +0.024 at ≥ 3, ≥ 5, ≥ 10). This was exploratory,
run to decide whether a hybrid is worth adding, and it isn't yet part of the protocol.

### Reported results (test split, run once)

| Model | Capped recall@10 | MRR@10 | nDCG@10 | Δ nDCG vs BM25 [95% CI] |
|---|---|---|---|---|
| BM25, stemmed (k₁ 0.4, b 1.0) | 0.233 | 0.354 | 0.215 | — |
| MiniLM | 0.177 | 0.282 | 0.163 | −0.051 [−0.060, −0.042] |
| mpnet | 0.181 | 0.296 | 0.170 | −0.044 [−0.055, −0.034] |
| BM25 + mpnet, RRF | pending | | | |

Test reproduces dev to within 0.002 nDCG, so the dev tuning didn't overfit.

## 10. Limitations so far

- **The labels reward wording.** Results measure agreement with a concordance-style
  reference, not "relatedness" in general. A thematic matcher is undervalued here.
- **Bible-only labels.** Book of Mormon retrieval is shown but not scored on public
  data. The private conference test supplies real Book of Mormon labels.
- **One translation.** All of this is KJV. Modern translations would likely narrow the
  embedding gap.
- **Off-the-shelf models.** Neither was fine-tuned on scripture. Fine-tuning on the
  cross-references would leak labels unless done with a strict split, and is out of
  scope.
