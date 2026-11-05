"""Religious Text Matcher: find scripture related to any passage.

    streamlit run app/app.py

Searches the public KJV Bible and Book of Mormon. With a local config.local.toml
(gitignored), it also searches private collections: the Doctrine and Covenants and
Pearl of Great Price, and a corpus of talks in docs/corpus-format.md format.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rtm.corpus import REPO_ROOT, load_corpus, load_local_config, load_scripture, load_scriptures_json
from rtm.search import BOOST_DELTA, Index

st.set_page_config(page_title="Religious Text Matcher", page_icon="📖", layout="wide")

METHOD_LABELS = {
    "hybrid": "Hybrid: keywords + meaning (best in evaluation)",
    "bm25": "Keywords (BM25)",
    "mpnet": "Meaning (mpnet)",
    "minilm": "Meaning, faster (MiniLM)",
}


@st.cache_resource(show_spinner="Loading scripture…")
def scripture_index() -> Index:
    frames = [load_scripture()]
    config = load_local_config() or {}
    private = config.get("private", {})
    for key, source, book_id in (("dc_json", "Doctrine and Covenants", "DC"),
                                 ("pgp_json", "Pearl of Great Price", None)):
        path = REPO_ROOT / private[key] if key in private else None
        if path and Path(path).exists():
            frames.append(load_scriptures_json(path, source, book_id))
    return Index(pd.concat(frames, ignore_index=True))


@st.cache_resource(show_spinner="Loading talks…")
def talk_index() -> Index | None:
    config = load_local_config() or {}
    path = config.get("corpus", {}).get("path")
    if not path or not Path(path).exists():
        return None
    return Index(load_corpus(path), chunk_docs=True)


def main() -> None:
    st.title("Religious Text Matcher")
    st.caption("Paste a passage, a paragraph from a talk, or a question, and find the "
               "scripture most related to it, by shared wording, by meaning, or both.")

    talks = talk_index()
    with st.sidebar:
        target = "Scripture verses"
        if talks is not None:
            name = (load_local_config() or {}).get("corpus", {}).get("name", "Talks")
            target = st.radio("Search in", ["Scripture verses", name])
        method = st.selectbox("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
        index = scripture_index() if target == "Scripture verses" else talks
        sources = sorted(index.docs["source"].unique())
        chosen = st.multiselect("Limit to", sources, placeholder="All sources")
        prefer_bom = False
        if target == "Scripture verses" and BOOST_DELTA:
            prefer_bom = st.toggle(
                "Prefer the Book of Mormon when close", value=True, disabled=method != "hybrid",
                help="Adds a small, evaluated margin to Book of Mormon verses in the hybrid "
                     "ranking, so a Book of Mormon parallel (2 Nephi 12) edges out its "
                     "source (Isaiah 2) when the two are near-equal.")
        k = st.slider("Results", 5, 50, 10, step=5)

    query = st.text_area("Text", height=160, placeholder="Charity suffereth long, and is kind…")
    if not query.strip():
        _about()
        return
    delta = BOOST_DELTA if prefer_bom and method == "hybrid" else 0.0
    with st.spinner("Searching…"):
        results = index.search(query, method=method, k=k, delta=delta, sources=chosen or None)
    for r in results.itertuples():
        heading = r.title if target == "Scripture verses" else f"{r.title}, {r.author} ({r.year})"
        with st.container(border=True):
            st.markdown(f"**{r.rank}. {heading}** · {r.source} · score {r.score:.3g}")
            text = r.text if len(r.text) <= 600 else r.text[:600].rsplit(" ", 1)[0] + "…"
            st.write(text)
            if target != "Scripture verses" and isinstance(getattr(r, "url", None), str):
                st.markdown(f"[Read the talk]({r.url})")
    _about()


def _about() -> None:
    with st.expander("How it works, and how well"):
        st.markdown(
            "- **Keywords (BM25)** finds verses that share distinctive words. **Meaning** "
            "models (sentence embeddings) find verses close in sense even with different "
            "words. **Hybrid** combines both rankings (reciprocal rank fusion).\n"
            "- Evaluated against about 600,000 crowd-voted Bible cross-references "
            "(OpenBible.info, CC BY) on 2,000 held-out verses: the hybrid ranked best "
            "(nDCG@10 0.226 vs 0.215 for keywords and 0.170 for meaning alone). Those "
            "cross-references reward shared wording, which favors keywords; see the "
            "project README for the full results and limitations.\n"
            "- Long texts are split into passages; each result is scored by its best "
            "match, so a 2,000-word talk isn't reduced to its first paragraph.\n"
            "- Looking for which talks cite a particular verse? The "
            "[BYU Scripture Citation Index](https://scriptures.byu.edu/) catalogs that. "
            "This tool answers the reverse question: given any text, including text no "
            "index has seen, which scripture relates to it by content.\n"
            "- Scripture text: King James Bible and Book of Mormon from Project Gutenberg "
            "(public domain). For study, not a doctrinal authority.")


main()
