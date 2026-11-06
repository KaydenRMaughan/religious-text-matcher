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
    "hybrid": "Hybrid: keywords + meaning",
    "bm25": "Keywords (BM25)",
    "mpnet": "Meaning (mpnet)",
    "minilm": "Meaning, faster (MiniLM)",
}

# What each method does, in plain terms, and what its score means.
METHOD_INFO = {
    "bm25": {
        "how": "Finds verses that share your words, weighting rare words (*charity*, *forgive*) "
               "far above common ones (*the*, *and*). Old and modern verb forms are matched "
               "(*loveth* = *love*).",
        "good": "Quoting or half-remembering a verse; distinctive names and terms.",
        "misses": "Same idea in different words. It can't tell that *anxious* relates to "
                  "*take no thought for the morrow*, and it rewards any shared word, even "
                  "when the verse means something else.",
        "score": "Keyword score: the summed weight of the matched words. It has no fixed scale "
                 "and only compares results within one search.",
    },
    "mpnet": {
        "how": "A sentence-embedding model turns your text and every verse into a vector that "
               "captures meaning, then finds the verses whose vectors point the same way.",
        "good": "Questions and feelings in modern words: *I feel anxious about the future*.",
        "misses": "Exact wording. It can rank a loosely related verse above the one you're "
                  "quoting, and it was trained on modern English, not KJV prose.",
        "score": "Cosine similarity, roughly 0 to 1; higher means closer in meaning. Long texts "
                 "are split into passages and each verse is scored against its best match.",
    },
    "minilm": {
        "how": "The same idea as mpnet with a smaller, faster model.",
        "good": "The same as mpnet, at lower cost.",
        "misses": "A bit less accurate than mpnet in the evaluation, with a shorter reading "
                  "window (256 tokens vs 384).",
        "score": "Cosine similarity, as for mpnet, but on its own scale: don't compare the two "
                 "models' numbers.",
    },
    "hybrid": {
        "how": "Runs Keywords and Meaning (mpnet) separately, then merges the two ranked lists "
               "(reciprocal rank fusion). A verse ranked high by both comes first.",
        "good": "Scripture-like text: a verse, a quoted passage, a paragraph from a talk. It "
                "scored best in the evaluation, where every query was a verse.",
        "misses": "Modern-language questions with no scriptural wording, where the keyword "
                  "half adds noise. Try Meaning for those.",
        "score": "Fused rank score: 1/(60 + rank) added across the two lists. About 0.033 means "
                 "both methods ranked it first; it measures agreement, not similarity.",
    },
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
    index = Index(pd.concat(frames, ignore_index=True))
    index.scores("warm up", "hybrid")  # load mpnet and the cached vectors now, not on the first search
    return index


@st.cache_resource(show_spinner="Loading talks…")
def talk_index() -> Index | None:
    config = load_local_config() or {}
    path = config.get("corpus", {}).get("path")
    if not path or not Path(path).exists():
        return None
    return Index(load_corpus(path), chunk_docs=True)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def _heading(r, verses: bool) -> str:
    return r.title if verses else f"{r.title}, {r.author} ({r.year})"


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
        compare = st.toggle("Compare all methods", value=False,
                            help="Run the same text through every method, side by side.")
        method = "hybrid"
        if not compare:
            method = st.selectbox("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
            info = METHOD_INFO[method]
            st.caption(f"{info['how']}\n\n**Good for:** {info['good']}\n\n"
                       f"**Misses:** {info['misses']}")
        index = scripture_index() if target == "Scripture verses" else talks
        sources = sorted(index.docs["source"].unique())
        chosen = st.multiselect("Limit to", sources, placeholder="All sources")
        prefer_bom = False
        if target == "Scripture verses" and BOOST_DELTA:
            prefer_bom = st.toggle(
                "Prefer the Book of Mormon when close", value=True,
                disabled=not compare and method != "hybrid",
                help="Hybrid only. Adds a small margin, sized on the known parallels, to Book "
                     "of Mormon verses, so a Book of Mormon parallel (Mosiah 14) edges out its "
                     "source (Isaiah 53) when the two are near-equal.")
        k = st.slider("Results", 5, 50, 10, step=5)

    query = st.text_area("Text", height=160, placeholder="Charity suffereth long, and is kind…")
    if not query.strip():
        _about()
        return
    verses = target == "Scripture verses"

    def run(m: str) -> pd.DataFrame:
        delta = BOOST_DELTA if prefer_bom and m == "hybrid" else 0.0
        return index.search(query, method=m, k=k, delta=delta, sources=chosen or None)

    with st.spinner("Searching…"):
        if compare:
            _compare({m: run(m) for m in METHOD_LABELS}, verses)
        else:
            _single(run(method), method, verses)
    _about()


def _single(results: pd.DataFrame, method: str, verses: bool) -> None:
    st.caption(f"Scores: {METHOD_INFO[method]['score']}")
    for r in results.itertuples():
        with st.container(border=True):
            st.markdown(f"**{r.rank}. {_heading(r, verses)}** · {r.source} · score {r.score:.3g}")
            st.write(_clip(r.text, 600))
            if not verses and isinstance(getattr(r, "url", None), str):
                st.markdown(f"[Read the talk]({r.url})")


def _compare(results: dict[str, pd.DataFrame], verses: bool) -> None:
    found = {m: set(r["id"]) for m, r in results.items()}
    in_all = set.intersection(*found.values())
    counts = pd.Series([i for ids in found.values() for i in ids]).value_counts()
    st.markdown(
        f"**{len(in_all)}** result(s) appear in every method's list. "
        "🟢 = found by all four · ⚪ = found by two or three · 🔸 = found by this method only. "
        "When the methods agree, the match is solid on both wording and meaning; where they "
        "split, see *Why do the methods give different results?* below. Scores can't be "
        "compared across columns.")
    cols = st.columns(len(results))
    for col, (m, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"#### {METHOD_LABELS[m]}")
            st.caption(METHOD_INFO[m]["score"].split(":")[0])
            for r in res.itertuples():
                n = counts[r.id]
                mark = "🟢" if n == len(results) else ("🔸" if n == 1 else "⚪")
                with st.container(border=True):
                    st.markdown(f"{mark} **{r.rank}. {_heading(r, verses)}**  \n"
                                f"<small>{r.source} · {r.score:.3g}</small>",
                                unsafe_allow_html=True)
                    st.caption(_clip(r.text, 160))


def _about() -> None:
    with st.expander("Why do the methods give different results?"):
        st.markdown(
            "Each method answers a different question, so they disagree most when your text "
            "and the scripture use different words for the same idea.\n\n"
            "- **Keywords** asks: *which verses share my words?* For *How can I forgive "
            "someone who hurt me?* it ranks Mark 2:7 first, \"who can forgive sins but God "
            "only?\" It shares *who can forgive* but answers a different question.\n"
            "- **Meaning** asks: *which verses say something similar?* For the same question "
            "it returns Mark 11:26 and 3 Nephi 13:14–15, on forgiving others. For *I feel "
            "anxious about the future*, it finds Mark 13:33, \"Take ye heed, watch and pray,\" "
            "while Keywords matches *feel* and lands on Isaac feeling Jacob's hands (Genesis "
            "27:12).\n"
            "- **Hybrid** asks: *which verses do both agree on?* It's strongest when your "
            "text sounds like scripture, and it inherits Keywords' noise on modern questions.\n"
            "- **When you quote scripture, they converge.** *Charity suffereth long* puts "
            "1 Corinthians 13:4 and Moroni 7:45 at the top of all four lists.\n"
            "- **The two Meaning models differ** because they're different networks trained "
            "differently. mpnet is larger and was slightly more accurate in testing.\n"
            "- **Scores aren't comparable across methods.** The same verse scored 28.3 "
            "(Keywords), 0.878 (mpnet) and 0.0328 (Hybrid). Compare ranks, not numbers.\n\n"
            "Turn on **Compare all methods** in the sidebar to see this for your own text.")
    with st.expander("How well does each method work?"):
        st.markdown(
            "- Evaluated against about 600,000 crowd-voted Bible cross-references "
            "(OpenBible.info, CC BY), using 2,000 held-out verses as queries. The hybrid "
            "ranked best (nDCG@10 0.226), then Keywords (0.215), mpnet (0.170) and MiniLM "
            "(0.163).\n"
            "- Those cross-references mostly link verses that share distinctive words, which "
            "favors Keywords. So the numbers say how well each method finds *traditionally "
            "cross-referenced* verses from *a scripture query*. They understate what Meaning "
            "adds for questions in your own words.\n"
            "- Long texts are split into passages; each result is scored by its best match, "
            "so a 2,000-word talk isn't reduced to its first paragraph. That made a large, "
            "measured difference on long documents.\n"
            "- Looking for which talks cite a particular verse? The "
            "[BYU Scripture Citation Index](https://scriptures.byu.edu/) catalogs that. "
            "This tool answers the reverse question: given any text, including text no "
            "index has seen, which scripture relates to it by content.\n"
            "- Scripture text: King James Bible and Book of Mormon from Project Gutenberg "
            "(public domain). For study, not a doctrinal authority.")


main()
