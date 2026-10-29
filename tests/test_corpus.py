import json

import pytest

from rtm import corpus
from rtm.corpus import CorpusError, load_corpus, load_cross_references, load_local_config


def parse(text, **kw):
    return corpus._parse_verses(text.split("\n"), **kw)


# --- verse parsing --------------------------------------------------------------

def test_mid_paragraph_verse_starts_are_split():
    verses = parse("1:1 First verse,\nwrapped. 1:2 Second verse.\n\n1:3 Third.")
    assert verses == [(1, 1, "First verse, wrapped."), (1, 2, "Second verse."), (1, 3, "Third.")]


def test_orphan_paragraph_continues_previous_verse():
    verses = parse("1:1 Many say of my soul, no help.\n\nSelah.\n\n1:2 Next.")
    assert verses[0] == (1, 1, "Many say of my soul, no help. Selah.")
    assert verses[1] == (1, 2, "Next.")


def test_continuation_with_embedded_verse_start():
    # Genesis 31:48-49 in the Gutenberg KJV: a stray blank line mid-verse.
    verses = parse("31:48 A witness this day.\n\nTherefore called Galeed; 31:49 And Mizpah.")
    assert verses == [(31, 48, "A witness this day. Therefore called Galeed;"), (31, 49, "And Mizpah.")]


def test_headers_and_chapter_labels_are_dropped():
    dropped = []
    text = "Otherwise Called:\n\nA book summary.\n\n1:1 One.\n\nBook Chapter 2\n\n2:1 Two.\n\n***"
    verses = parse(text, dropped=dropped)
    assert verses == [(1, 1, "One."), (2, 1, "Two.")]
    assert dropped == ["Otherwise Called:", "A book summary.", "Book Chapter 2", "***"]


def test_reference_prefix_is_stripped():
    verses = parse("4 Nephi 1:1 First.\n\n4 Nephi 1:2 Second.", ref_prefix="4 Nephi")
    assert verses == [(1, 1, "First."), (1, 2, "Second.")]


def test_order_check_catches_a_skipped_verse():
    with pytest.raises(CorpusError, match="1:1 followed by 1:3"):
        corpus._check_order("Gen", [(1, 1, "a"), (1, 3, "b")])


# --- cross-references ---------------------------------------------------------------

def test_cross_reference_ranges_expand_and_unknown_ids_drop(tmp_path):
    path = tmp_path / "xref.txt"
    path.write_text(
        "From Verse\tTo Verse\tVotes\t#attribution\n"
        "Gen.1.1\tGen.1.2-Gen.1.3\t5\n"
        "Gen.1.2\tGen.9.9\t1\n"
        "Gen.1.3\tGen.1.1\t-2\n"
    )
    df = load_cross_references(path, ["Gen.1.1", "Gen.1.2", "Gen.1.3"])
    assert df.to_dict("records") == [
        {"from_id": "Gen.1.1", "to_id": "Gen.1.2", "votes": 5},
        {"from_id": "Gen.1.1", "to_id": "Gen.1.3", "votes": 5},
        {"from_id": "Gen.1.3", "to_id": "Gen.1.1", "votes": -2},
    ]


# --- generic corpus format ------------------------------------------------------------

def test_load_corpus_json_and_jsonl(tmp_path):
    docs = [{"id": 1, "title": "A", "text": "alpha", "year": "2020"}, {"id": "2", "title": "B", "text": "beta"}]
    (tmp_path / "c.json").write_text(json.dumps(docs))
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    for name in ("c.json", "c.jsonl"):
        df = load_corpus(tmp_path / name)
        assert df["id"].tolist() == ["1", "2"]
        assert df["year"].tolist()[0] == 2020
        assert {"source", "author", "url"} <= set(df.columns)


@pytest.mark.parametrize(
    "docs, message",
    [
        ([{"id": "1", "title": "A"}], "missing"),
        ([{"id": "1", "title": "A", "text": "x"}, {"id": "1", "title": "B", "text": "y"}], "duplicate"),
        ({"id": "1"}, "JSON array"),
    ],
)
def test_load_corpus_rejects_bad_input(tmp_path, docs, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(docs))
    with pytest.raises(CorpusError, match=message):
        load_corpus(path)


def test_local_config_resolves_relative_path(tmp_path):
    assert load_local_config(tmp_path) is None
    (tmp_path / "config.local.toml").write_text('[corpus]\npath = "data/private/c.json"\nname = "Mine"\n')
    config = load_local_config(tmp_path)
    assert config["corpus"]["path"] == tmp_path / "data/private/c.json"
    assert config["corpus"]["name"] == "Mine"


# --- the real demo corpus (skipped until scripts/fetch_data.py has run) --------------

needs_data = pytest.mark.skipif(
    not (corpus.RAW_DIR / "kjv.txt").exists(), reason="run scripts/fetch_data.py first"
)


@needs_data
def test_scripture_verse_counts():
    df = corpus.load_scripture()
    counts = df["source"].value_counts().to_dict()
    # Standard counts: KJV 31,102 verses; Book of Mormon 6,604.
    assert counts == {"Old Testament": 23145, "New Testament": 7957, "Book of Mormon": 6604}
    assert df["id"].is_unique
    assert not df["text"].str.contains(r"\d+:\d+|Chapter \d").any()


@needs_data
def test_cross_references_join_to_kjv():
    df = corpus.load_scripture()
    kjv_ids = df.loc[df["source"] != "Book of Mormon", "id"].tolist()
    xref = load_cross_references(corpus.RAW_DIR / "cross_references.txt", kjv_ids)
    assert len(xref) > 600_000
    assert xref["from_id"].nunique() > 29_000
