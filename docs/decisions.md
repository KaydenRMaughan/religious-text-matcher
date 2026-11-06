# Decisions

## 2026-10-29: Kickoff

**Scope.** S tier, 10–12 hrs, ships by Dec 20 2026. Rebuilt from a private prototype (a General Conference talk matcher) as a fresh repo with no carried-over history.

**Public and private versions.**
- *Public (this repo):* a corpus-agnostic matcher, a public-domain scripture pipeline, an evaluation harness, and a loader for any corpus in [corpus-format.md](corpus-format.md).
- *Private (local only, never pushed):* the General Conference corpus and its collection code live outside this repo. The app reads them through a gitignored `config.local.toml`. No conference text, scraper, or derived embeddings are ever committed here.
- *Why:* churchofjesuschrist.org's Terms of Use prohibit automated access and reposting material without permission, so neither the corpus nor the collection code can be published.

**Demo corpus: KJV and Book of Mormon (Project Gutenberg).** Both are public domain in the US. Downloaded by script rather than committed, which avoids any question about redistributing the files.

**Evaluation labels: OpenBible.info cross-references (CC BY).** About 340k verse-to-verse links, mostly from the public-domain Treasury of Scripture Knowledge. They cover the Bible only, so Book of Mormon retrieval is demoed but not scored.

**Baseline: BM25.** Embeddings have to beat keyword search to earn their place.

**Chunking over truncation.** The prototype embedded whole talks with `all-MiniLM-L6-v2`, which truncates at 256 tokens. Median talk length was ~1,700 words, so only the opening ~10–15% of each talk was matched. Texts are now split into passages, and a document scores as its best passage (max-pooling). Chunking is evaluated against truncation rather than assumed better.

**No FAISS.** About 37k verses × 384–768 dimensions fits in memory, and exact cosine search with NumPy is fast enough. Revisit above ~1M passages.

**Dropped from the prototype:**
- Jenks similarity tiers: the display logic never triggered, and the tiers implied more precision than scores support.
- Unused TensorFlow and UMAP imports.
- PCA refit on every query: the projection is now fit once at load.

**Out of scope:** RAG and LLM generation (that's the separate Conference Compass idea), UMAP, and any further scraping.

## 2026-10-29: Data pipeline

**Requirements pinned on Python 3.14** (Windows, CPU). Direct dependencies only, plus `torch`, which is the heaviest and most version-sensitive transitive dependency. The package installs editable (`pip install -e .`) from a `src/` layout.

**Fetch, don't commit.** `scripts/fetch_data.py` downloads the two Gutenberg texts and the OpenBible.info zip into `data/raw/`, skips files already present, waits a second between hosts' requests, and writes `SOURCES.json` with URL, date and SHA-256 so a later run can tell whether the upstream files changed.

**Verse parsing rules** (`src/rtm/corpus.py`), learned from the files rather than assumed:
- Verses are paragraphs starting `c:v`, but some verses start mid-paragraph, so each paragraph is split on the `c:v` pattern.
- The source often breaks a verse's last sentence onto its own paragraph ("Selah.", "Amen."; Genesis 31:48 has a stray break mid-verse). A paragraph that doesn't start with `c:v` therefore attaches to the previous verse. The first draft dropped these and would have truncated 114 verses.
- Dropped: anything before a book's first verse (titles, "Otherwise Called:", Book of Mormon book summaries), chapter labels ("1 Nephi Chapter 2"), and the `***` / New Testament separator.
- 4 Nephi prints the book name before each reference ("4 Nephi 1:1"); that known prefix is stripped explicitly, not by a general rule that could eat a proper noun ending a KJV verse.
- Every book must start at 1:1 and run in order, or loading fails. That check caught both bugs above.
- Result: 31,102 KJV and 6,604 Book of Mormon verses, matching the standard counts. This edition has no Psalm superscriptions.

**Cross-references.** Range targets (`Gen.1.2-Gen.1.3`) expand to one row per verse: 344,799 links become 614,179 verse pairs from 29,363 source verses. Five links to 3 John 1:15 drop because this edition folds that verse into 1:14. 3,512 links have zero or negative votes. The vote threshold is an evaluation-harness decision, not a loader one.

## 2026-10-30: Evaluation protocol

Settled by Kayden after a walkthrough of what each choice does to the models and to the measurement. Full protocol in [how-it-works.md](how-it-works.md#evaluation-protocol-settled-2026-10-30).

**Votes ≥ 5 as the relevance cut**, with ≥ 3 and ≥ 10 as robustness rows. ≥ 1 gives a median of 15 relevant verses per query, many weak; ≥ 10 keeps only 7.7k queries, skewed toward famous passages.

**Binary relevance primary, graded secondary.** Votes measure popularity as well as relevance, so graded gain would reward models that favor famous verses.

**Dev and test query sets** (2,000 each, seeded, stratified by testament). Tuning BM25 on the reported queries would overstate the baseline.

**Neighbors stay in.** The first framing (that adjacent verses inflate scores) was wrong: only 0.2% of links go to an adjacent verse, so neighbors cost lexical models top-10 slots rather than giving free hits. The query verse and 66 self-links are removed.

**Capped and uncapped recall both reported.**

**Chunking is tested twice:** on the private conference corpus against each talk's own scripture citations (results stay private), and on a public second task over chapters. Design and budget still open.

**BM25 uses Lucene's IDF**, not rank-bm25's Okapi default: `and` is in 79% of verses, where Okapi's IDF is negative and rank-bm25 floors it ad hoc.

**Tokenizer: stemmed and unstemmed variants, chosen on dev.** Snowball doesn't touch KJV -eth/-est forms (`loveth`, `cometh`), so the stemmed variant maps them to modern forms first, without touching `priest`, `rest` or `Seth`. An unstemmed-only BM25 would be a straw man against the embedding models. Adds `snowballstemmer` (pure Python, BSD).

**Book of Mormon verses are eligible and preferred answers.** This supersedes the kickoff's "demoed but not scored." How unjudged Book of Mormon results are scored, and what "preferred" means, are still open.

**Follow-up, same day.** D&C is added privately for the conference test. Unjudged Book of Mormon results are dropped before scoring (condensed lists). "Preferred" means a score boost that wins when scores are close (2 Nephi 12 over Isaiah 2); its form and size are set at the app stage.

## 2026-10-30: BM25 baseline is stemmed

On the 2,000 dev queries at votes ≥ 5, stemmed BM25 beats plain on nDCG@10 by +0.0056 (paired bootstrap 95% interval +0.0014 to +0.0098). It's ahead on every metric at every cut, but only that interval excludes 0, and it is worse on more queries than it is better (318 better, 388 worse on nDCG): a few big wins and many small losses, the usual stemming pattern. Kayden chose stemmed: never meaningfully worse, and the stronger baseline is the conservative one to measure embeddings against.

Dev BM25 (stemmed, k₁ = 1.5, b = 0.75, untuned), votes ≥ 5: Recall@10 0.201, capped 0.225, MRR@10 0.353, nDCG@10 0.211; about 11% of the raw top 10 is Book of Mormon.

## 2026-11-05: Dev results; embeddings lose alone, a hybrid wins

Run on the 2,000 dev queries with the Tutorial 3 functions (drafted by Claude at Kayden's request; walkthrough in [methods.md](methods.md)).

**BM25 tuning** (35 combinations, nDCG@10 at votes ≥ 5): best k₁ = 0.4, b = 1.0, +0.005 over the defaults (interval +0.002 to +0.008). The surface is flat near the top. Choice pending Kayden's confirmation.

**Embeddings alone lose to tuned BM25** at every cut (mpnet −0.045 nDCG at ≥ 5, interval −0.055 to −0.034; MiniLM −0.053). Diagnosed: 47% of labeled links share a rare stemmed word vs 4% of random pairs, because the labels descend from a concordance-style reference. Also, the models were trained on modern English. On links with no shared rare word, mpnet's recall is 2.3x BM25's, but all models are near zero.

**Exploratory hybrid:** reciprocal rank fusion of BM25 and mpnet beats BM25 at every cut, with every interval excluding 0. Whether to add it as a fourth model is Kayden's call.

**D&C text for the private test.** From [bcbooks/scriptures-json](https://github.com/bcbooks/scriptures-json) (`doctrine-and-covenants.json`, version dc-3), which its README declares public domain. It leaves out the copyrighted introduction and the official declarations (OD 2 dates from 1978). The file carries no section headings or footnotes. It's loaded with `load_scriptures_json` (ids `DC.27.15`), giving 138 sections and 3,654 verses, the standard count. It sits in gitignored `data/private/` and is never committed. Its terms would allow publishing it, but it's used only for the private test, so it stays local.

**Citation parser** (`src/rtm/citations.py`, Claude's). It handles full names and Church-style abbreviations, ranges (including ranges across chapters), comma lists, `section 89`, `Alma chapter 32`, and bare `36:16-19`, which takes the previous citation's book. Dashes stored as U+FFFD are read as ranges. Chapter-only citations come back separately from verse citations, so the protocol can choose what to do with them. Prose citations ("the eighth chapter of Moroni") and the Joseph Smith Translation stay unparsed. On the private corpus, 2,211 of 2,327 talks cite at least one parsable verse (median 14). The 50,229 cited verse ids split into Bible 22,983, Book of Mormon 16,526 and D&C 10,720. Pearl of Great Price citations (about 1,600) need that text loaded too.

## 2026-11-05: Final settings; reported test run (three models)

**Confirmed by Kayden:** BM25 k₁ = 0.4, b = 1.0 (now the `run_eval.py` defaults). Also: add the BM25 + mpnet RRF hybrid as a fourth model (`rrf_scores`, Kayden's to type, c = 60 fixed), and add the Pearl of Great Price privately (same public-domain source; 635 verses, standard counts; 2,831 more cited verses).

**Test split, run once** (2,000 queries; nDCG@10, votes ≥ 5): BM25 0.215, MiniLM 0.163, mpnet 0.170. Paired differences against BM25: MiniLM −0.051 [−0.060, −0.042], mpnet −0.044 [−0.055, −0.034]. Every embedding deficit, at every cut and on every metric, has an interval that excludes 0. These closely reproduce dev, which suggests the dev tuning didn't overfit. The hybrid's test run follows once `rrf_scores` is typed.

**Hybrid, test split** (`rrf_scores` typed by Kayden): nDCG@10 0.226 vs BM25's 0.215 at votes ≥ 5, a difference of +0.011 [+0.004, +0.019]. Capped recall +0.018 [+0.008, +0.027]. Both are positive at every cut. MRR is positive at every cut but its interval excludes 0 only at ≥ 10. **The headline: embeddings alone lose to BM25 on these labels, and the hybrid beats both.**

## 2026-11-05: BYU Scripture Citation Index: acknowledged, not used

Kayden decided not to use [scriptures.byu.edu](https://scriptures.byu.edu/) as data: it states no terms of use and offers no bulk access, and its citations come from the same talk footnotes our labels already use. The README and app must acknowledge it, though. It's the established tool for looking up which talks cite a given verse. This project does the reverse: given any passage, including new text that no index has seen, it finds related scripture by content.

## 2026-11-05: Chapter task results (public, test split)

Verse queries against the 1,189 Bible chapters (median 627 words). A chapter is relevant if it holds a ≥ 5-vote target, excluding the query's own chapter. 1,956 test queries at ≥ 5.

**Chunking beats truncation for both models, at every cut** (nDCG@10 at ≥ 5): MiniLM 0.102 → 0.155, +0.054 [+0.045, +0.063]. mpnet 0.122 → 0.160, +0.038 [+0.029, +0.047]. MiniLM gains more because its 256-token window truncates more of each chapter than mpnet's 384. This confirms the prototype audit's main finding with public data.

**BM25 is stronger still on chapters** (0.264). The hybrid doesn't beat it here: −0.014 [−0.024, −0.005] at ≥ 5, −0.012 at ≥ 3, and +0.007 (interval spans 0) at ≥ 10. With a partner this much weaker, equal-weight fusion costs a little. The hybrid's advantage holds for verse retrieval, not chapter retrieval. The README states both.

## 2026-11-06: Book of Mormon boost set to 0.0006 (decision 8b closed)

**Form:** additive, on the hybrid's RRF scores, after fusion. **Size: δ = 0.0006**, chosen by Kayden from the known parallels only (`scripts/boost_parallels.py`, public data).

The parallel chapters (2 Ne 12–24 ↔ Isa 2–14, 1 Ne 20–21 ↔ Isa 48–49, Mosiah 14 ↔ Isa 53, 3 Ne 22 ↔ Isa 54, 3 Ne 12–14 ↔ Matt 5–7) align into 445 verse pairs on stemmed-token Jaccard ≥ 0.5. The median Jaccard is 1.00: most are word-for-word twins. The test cases are 4,468 (query, pair) cases, using queries OpenBible links (≥ 5 votes) to the Bible side, with queries from the parallel chapters excluded.

| δ | Book of Mormon ahead of its source | …when the source is in the top 10 | Book of Mormon share of the top 10 (public dev) |
|---|---|---|---|
| 0 | 43% | 30% | 11.1% |
| 0.0003 | 80% | 46% | 12.0% |
| 0.0005 | 85% | 58% | 12.6% |
| 0.001 | 90% | 72% | 14.2% |
| 0.002 | 94% | 83% | 17.6% |

**Why so small a margin still matters.** Twins get identical BM25 and cosine scores, so the tie is broken by corpus order and the Bible verse wins. RRF then turns the tie into adjacent ranks in both lists, and flipping a pair at rank r takes δ > 2 / ((60 + r)(61 + r)): about 0.00054 at rank 1 and 0.0004 at rank 10. At 0.0006, twins flip anywhere in the top 10, and the Book of Mormon share of the top 10 rises by about 1.5 points.

**Private dev, for the record (not used to choose):** every δ > 0 lowered nDCG@10 on the talk-citation labels (best at δ = 0). The preference is a stated product choice, not something either label set rewards.

**Alternative considered:** a tie-break before fusion (no flooding at all, but near-twins not guaranteed to flip). Not taken.

## 2026-11-06: App launched; README rewritten

- **App:** `streamlit run app/app.py`. The hybrid is the default, and the boost toggle is on by default for the hybrid only. The model warm-up moved into the cached index load (cold start about 27 s; searches then take under 0.1 s). The private talks search reuses the eval's cached mpnet passage vectors (24,561 passages, cache hit), so it doesn't re-encode.
- **README:** rewritten around the results. Embeddings alone lose to BM25 on these labels and the hybrid wins on verses. Chunking wins on long documents, where BM25 leads. The private test is described by method only, with no numbers. The BYU Scripture Citation Index is acknowledged. The old "meaning, not keywords" framing is gone.
