# Religious Text Matcher

Paste a passage, a study note, or a personal reflection and find the scripture closest to it **in meaning**, not keywords.

> **Status:** in development (target: December 2026). Results below are placeholders until the evaluation runs.

## What it does

Sentence-embedding semantic search over the King James Bible and the Book of Mormon, with a BM25 keyword baseline for comparison. Long texts are split into passages before embedding, so matches reflect the whole text and not just its opening lines.

## Why it's built this way

- **Evaluated, not just demoed.** Retrieval quality is measured against ~340k human-curated cross-references (OpenBible.info): Recall@10, MRR, and nDCG@10, for BM25 vs. two embedding models, with and without chunking.
- **Corpus-agnostic.** Any collection of texts in a simple JSON format can be loaded locally (see [docs/corpus-format.md](docs/corpus-format.md)). The public demo uses public-domain text only.

## Data

| Source | Use | License |
|---|---|---|
| King James Version (Project Gutenberg #10) | Demo corpus | Public domain (US) |
| Book of Mormon (Project Gutenberg #17) | Demo corpus | Public domain (US) |
| [OpenBible.info cross-references](https://www.openbible.info/labs/cross-references/) | Evaluation labels | CC BY |

Texts are downloaded by `scripts/fetch_data.py` into `data/raw/` (gitignored), which also records each file's URL, date and SHA-256 in `data/raw/SOURCES.json`. No corpus text is committed to this repo.

The parser yields one record per verse: 31,102 for the KJV and 6,604 for the Book of Mormon, the standard counts. Bible verse ids (`Gen.1.1`) match the cross-reference ids, and all but 5 of 344,799 cross-reference links join. The 5 point to 3 John 1:15, which this edition folds into 1:14.

## Results

_TBD._

## Limitations

- A similarity score is a relative ranking signal, not a probability or a measure of doctrinal relevance.
- Cross-references are an imperfect label: a sensible match the references don't list counts as a miss.
- Study aid only. It carries no doctrinal authority.
- If you deploy your own copy, don't paste private or sacred personal text into a hosted instance. Run it locally instead.

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -r requirements.txt
pip install -e .
python scripts/fetch_data.py
streamlit run app/app.py
```

## Stack

Python · sentence-transformers · rank-bm25 · NumPy · pandas · Streamlit · Plotly

## License

Code: MIT. Corpus texts and evaluation data keep their own licenses (above).
