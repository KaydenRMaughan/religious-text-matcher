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
