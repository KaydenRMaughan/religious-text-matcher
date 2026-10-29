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
