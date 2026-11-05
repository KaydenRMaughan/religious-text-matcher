"""Corpus loaders.

Two kinds of input:

- The public demo corpus: Project Gutenberg plain-text editions of the King James
  Bible (#10) and the Book of Mormon (#17), parsed into one record per verse.
  Bible verse ids use the OSIS-style abbreviations OpenBible.info uses
  (``Gen.1.1``), so they join directly to the cross-reference labels.
- Any local corpus in the JSON / JSON Lines format in docs/corpus-format.md,
  optionally named in a gitignored ``config.local.toml``.

Every loader returns a pandas DataFrame with at least ``id``, ``title``, ``text``
and ``source`` columns, plus the optional ``year``, ``author`` and ``url``.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

REQUIRED_FIELDS = ("id", "title", "text")
OPTIONAL_FIELDS = ("source", "year", "author", "url")

# (OSIS id, display name), in canonical order. Ids match OpenBible.info.
BIBLE_BOOKS = [
    ("Gen", "Genesis"), ("Exod", "Exodus"), ("Lev", "Leviticus"), ("Num", "Numbers"),
    ("Deut", "Deuteronomy"), ("Josh", "Joshua"), ("Judg", "Judges"), ("Ruth", "Ruth"),
    ("1Sam", "1 Samuel"), ("2Sam", "2 Samuel"), ("1Kgs", "1 Kings"), ("2Kgs", "2 Kings"),
    ("1Chr", "1 Chronicles"), ("2Chr", "2 Chronicles"), ("Ezra", "Ezra"),
    ("Neh", "Nehemiah"), ("Esth", "Esther"), ("Job", "Job"), ("Ps", "Psalms"),
    ("Prov", "Proverbs"), ("Eccl", "Ecclesiastes"), ("Song", "Song of Solomon"),
    ("Isa", "Isaiah"), ("Jer", "Jeremiah"), ("Lam", "Lamentations"), ("Ezek", "Ezekiel"),
    ("Dan", "Daniel"), ("Hos", "Hosea"), ("Joel", "Joel"), ("Amos", "Amos"),
    ("Obad", "Obadiah"), ("Jonah", "Jonah"), ("Mic", "Micah"), ("Nah", "Nahum"),
    ("Hab", "Habakkuk"), ("Zeph", "Zephaniah"), ("Hag", "Haggai"), ("Zech", "Zechariah"),
    ("Mal", "Malachi"),
    ("Matt", "Matthew"), ("Mark", "Mark"), ("Luke", "Luke"), ("John", "John"),
    ("Acts", "Acts"), ("Rom", "Romans"), ("1Cor", "1 Corinthians"),
    ("2Cor", "2 Corinthians"), ("Gal", "Galatians"), ("Eph", "Ephesians"),
    ("Phil", "Philippians"), ("Col", "Colossians"), ("1Thess", "1 Thessalonians"),
    ("2Thess", "2 Thessalonians"), ("1Tim", "1 Timothy"), ("2Tim", "2 Timothy"),
    ("Titus", "Titus"), ("Phlm", "Philemon"), ("Heb", "Hebrews"), ("Jas", "James"),
    ("1Pet", "1 Peter"), ("2Pet", "2 Peter"), ("1John", "1 John"), ("2John", "2 John"),
    ("3John", "3 John"), ("Jude", "Jude"), ("Rev", "Revelation"),
]
NEW_TESTAMENT_START = 39  # index of Matthew

# (id, display name, heading prefix in the Gutenberg text), in book order.
# The display name doubles as the reference prefix 4 Nephi prints before verses.
BOOK_OF_MORMON_BOOKS = [
    ("1Ne", "1 Nephi", "THE FIRST BOOK OF NEPHI"),
    ("2Ne", "2 Nephi", "THE SECOND BOOK OF NEPHI"),
    ("Jacob", "Jacob", "THE BOOK OF JACOB"),
    ("Enos", "Enos", "THE BOOK OF ENOS"),
    ("Jarom", "Jarom", "THE BOOK OF JAROM"),
    ("Omni", "Omni", "THE BOOK OF OMNI"),
    ("WofM", "Words of Mormon", "THE WORDS OF MORMON"),
    ("Mosiah", "Mosiah", "THE BOOK OF MOSIAH"),
    ("Alma", "Alma", "THE BOOK OF ALMA"),
    ("Hel", "Helaman", "THE BOOK OF HELAMAN"),
    ("3Ne", "3 Nephi", "THIRD BOOK OF NEPHI"),
    ("4Ne", "4 Nephi", "FOURTH NEPHI"),
    ("Morm", "Mormon", "THE BOOK OF MORMON"),
    ("Ether", "Ether", "THE BOOK OF ETHER"),
    ("Moro", "Moroni", "THE BOOK OF MORONI"),
]

_VERSE_START = re.compile(r"(?<!\S)(\d+):(\d+)\s+")
_PARAGRAPH_VERSE = re.compile(r"^\d+:\d+\s")
# Paragraphs after a book's first verse that aren't verse text.
_NON_VERSE = re.compile(r"^(?:.+ Chapter \d+|\*+|The New Testament of the King James Bible)$")


class CorpusError(ValueError):
    """Raised when a corpus file is malformed or doesn't parse as expected."""


# --- Project Gutenberg plain text -------------------------------------------

def _gutenberg_body(raw: str) -> list[str]:
    """Return the lines between the Gutenberg START and END markers."""
    lines = raw.replace("\r\n", "\n").split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("*** START OF")), None)
    end = next((i for i, l in enumerate(lines) if l.startswith("*** END OF")), None)
    if start is None or end is None:
        raise CorpusError("Project Gutenberg START/END markers not found")
    return lines[start + 1 : end]


def _parse_verses(
    lines: list[str], ref_prefix: str | None = None, dropped: list[str] | None = None
) -> list[tuple[int, int, str]]:
    """Parse one book's lines into (chapter, verse, text).

    Verses sit in blank-line-separated paragraphs that start with ``c:v``; a few
    verses also start mid-paragraph. Text before a book's first verse (titles,
    book summaries) is dropped, as are chapter labels and section separators.
    Any other paragraph that doesn't start with ``c:v`` continues the previous
    verse: the source often breaks a verse's last sentence onto its own
    paragraph ("Selah.", Genesis 31:48). ``dropped``, if given, collects what was
    dropped. ``ref_prefix`` strips a book name printed before each reference
    (``4 Nephi 1:1``).
    """
    paragraphs = re.split(r"\n\s*\n", "\n".join(lines))
    prefix_re = (
        re.compile(rf"(?<!\S){re.escape(ref_prefix)}\s+(?=\d+:\d+\s)") if ref_prefix else None
    )
    verses: list[list] = []
    for para in paragraphs:
        para = " ".join(para.split())
        if prefix_re:
            para = prefix_re.sub("", para)
        if not para:
            continue
        parts = _VERSE_START.split(para)  # [lead, c, v, text, c, v, text, ...]
        lead = parts[0].strip()
        if lead:
            if not verses or _NON_VERSE.match(lead):
                if dropped is not None:
                    dropped.append(para)
                continue
            verses[-1][2] += " " + lead
        for i in range(1, len(parts), 3):
            verses.append([int(parts[i]), int(parts[i + 1]), parts[i + 2].strip()])
    return [(c, v, text) for c, v, text in verses]


def _check_order(book: str, verses: list[tuple[int, int, str]]) -> None:
    """Fail loudly if a book's verses don't start at 1:1 and run in order."""
    if not verses or verses[0][:2] != (1, 1):
        raise CorpusError(f"{book}: does not start at 1:1")
    for (c0, v0, _), (c1, v1, _) in zip(verses, verses[1:]):
        if not ((c1 == c0 and v1 == v0 + 1) or (c1 == c0 + 1 and v1 == 1)):
            raise CorpusError(f"{book}: {c0}:{v0} followed by {c1}:{v1}")


def _split_books(body: list[str], starts: list[int], names: list[str]) -> list[list[str]]:
    if len(starts) != len(names):
        raise CorpusError(f"found {len(starts)} book headings, expected {len(names)}")
    bounds = starts + [len(body)]
    return [body[bounds[i] + 1 : bounds[i + 1]] for i in range(len(names))]


def parse_kjv(raw: str) -> pd.DataFrame:
    """Parse the Gutenberg KJV (#10) into one row per verse."""
    body = _gutenberg_body(raw)
    # The table of contents lists the 66 book titles exactly as the body headings
    # spell them. The Old Testament header appears at the top of the contents and
    # again where the text begins; the last one before the first verse is the latter.
    first_verse = next((i for i, l in enumerate(body) if _PARAGRAPH_VERSE.match(l)), None)
    ot_headers = [
        i for i, l in enumerate(body[:first_verse])
        if l.strip().startswith("The Old Testament of the King James")
    ]
    if first_verse is None or not ot_headers:
        raise CorpusError("KJV: Old Testament header not found")
    ot_header = ot_headers[-1]
    toc = [
        l.strip() for l in body[:ot_header]
        if l.strip() and not l.strip().startswith(("The Old Testament", "The New Testament", "The King James"))
    ]
    if len(toc) != len(BIBLE_BOOKS):
        raise CorpusError(f"KJV: table of contents lists {len(toc)} books, expected {len(BIBLE_BOOKS)}")

    starts, search_from = [], ot_header + 1
    for title in toc:
        idx = next((i for i in range(search_from, len(body)) if body[i].strip() == title), None)
        if idx is None:
            raise CorpusError(f"KJV: heading not found: {title!r}")
        starts.append(idx)
        search_from = idx + 1

    rows = []
    for b, book_lines in enumerate(_split_books(body, starts, toc)):
        osis, name = BIBLE_BOOKS[b]
        verses = _parse_verses(book_lines)
        _check_order(osis, verses)
        source = "New Testament" if b >= NEW_TESTAMENT_START else "Old Testament"
        rows += [
            {"id": f"{osis}.{c}.{v}", "title": f"{name} {c}:{v}", "text": text, "source": source,
             "book": name, "chapter": c, "verse": v}
            for c, v, text in verses
        ]
    return pd.DataFrame(rows)


def parse_book_of_mormon(raw: str) -> pd.DataFrame:
    """Parse the Gutenberg Book of Mormon (#17) into one row per verse."""
    body = _gutenberg_body(raw)
    contents = next((i for i, l in enumerate(body) if l.strip() == "Contents"), None)
    if contents is None:
        raise CorpusError("Book of Mormon: Contents block not found")
    # Skip the contents list itself: it ends at the first blank line after it.
    search_from = next(i for i in range(contents + 2, len(body)) if not body[i].strip())

    starts = []
    for _, _, heading in BOOK_OF_MORMON_BOOKS:
        idx = next(
            (i for i in range(search_from, len(body)) if body[i].strip().startswith(heading)), None
        )
        if idx is None:
            raise CorpusError(f"Book of Mormon: heading not found: {heading!r}")
        starts.append(idx)
        search_from = idx + 1

    rows = []
    names = [h for _, _, h in BOOK_OF_MORMON_BOOKS]
    for b, book_lines in enumerate(_split_books(body, starts, names)):
        book_id, name, _ = BOOK_OF_MORMON_BOOKS[b]
        verses = _parse_verses(book_lines, ref_prefix=name)
        _check_order(book_id, verses)
        rows += [
            {"id": f"{book_id}.{c}.{v}", "title": f"{name} {c}:{v}", "text": text,
             "source": "Book of Mormon", "book": name, "chapter": c, "verse": v}
            for c, v, text in verses
        ]
    return pd.DataFrame(rows)


def load_scripture(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the public demo corpus (KJV + Book of Mormon) from data/raw/."""
    paths = {"kjv": raw_dir / "kjv.txt", "bom": raw_dir / "book_of_mormon.txt"}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {raw_dir}; run scripts/fetch_data.py")
    kjv = parse_kjv(paths["kjv"].read_text(encoding="utf-8"))
    bom = parse_book_of_mormon(paths["bom"].read_text(encoding="utf-8"))
    return pd.concat([kjv, bom], ignore_index=True)


# --- Evaluation labels --------------------------------------------------------

def load_cross_references(path: Path, verse_order: list[str]) -> pd.DataFrame:
    """Load OpenBible.info cross-references as (from_id, to_id, votes).

    Range targets such as ``Gen.1.1-Gen.1.3`` are expanded to one row per verse
    using ``verse_order`` (the KJV ids in canonical order). Rows naming a verse
    absent from ``verse_order`` are dropped.
    """
    # The header row carries a fourth, attribution-only field; skip it.
    df = pd.read_csv(path, sep="\t", skiprows=1, header=None, usecols=[0, 1, 2],
                     names=["from_id", "to_ref", "votes"], dtype={"votes": "int64"})
    position = {vid: i for i, vid in enumerate(verse_order)}

    def expand(ref: str) -> list[str]:
        first, _, last = ref.partition("-")
        last = last or first
        if first not in position or last not in position or position[last] < position[first]:
            return []
        return verse_order[position[first] : position[last] + 1]

    df["to_id"] = df["to_ref"].map(expand)
    df = df.explode("to_id").dropna(subset=["to_id"])
    df = df[df["from_id"].isin(position)]
    return df[["from_id", "to_id", "votes"]].reset_index(drop=True)


# --- Generic local corpora ------------------------------------------------------

def load_corpus(path: Path) -> pd.DataFrame:
    """Load a corpus in the docs/corpus-format.md format (JSON array or JSON Lines)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if not isinstance(records, list):
        raise CorpusError(f"{path.name}: expected a JSON array of documents")

    for n, rec in enumerate(records):
        missing = [f for f in REQUIRED_FIELDS if not rec.get(f)]
        if missing:
            raise CorpusError(f"{path.name}: document {n} is missing {missing}")

    df = pd.DataFrame(records)
    df["id"] = df["id"].astype(str)
    dupes = df.loc[df["id"].duplicated(), "id"].unique()
    if len(dupes):
        raise CorpusError(f"{path.name}: duplicate ids, e.g. {list(dupes[:5])}")
    for field in OPTIONAL_FIELDS:
        if field not in df:
            df[field] = None
    if df["year"].notna().any():
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype("Int64")
    return df


def load_local_config(repo_root: Path = REPO_ROOT) -> dict | None:
    """Read config.local.toml (gitignored) if present; None otherwise.

    A relative ``[corpus] path`` is resolved against the repo root.
    """
    config_path = repo_root / "config.local.toml"
    if not config_path.exists():
        return None
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    corpus = config.get("corpus", {})
    if "path" in corpus:
        p = Path(corpus["path"])
        corpus["path"] = p if p.is_absolute() else repo_root / p
    return config


# lds_slug in bcbooks/scriptures-json -> corpus book id (Pearl of Great Price).
PEARL_BOOKS = {"moses": "Moses", "abr": "Abr", "js-m": "JSM", "js-h": "JSH", "a-of-f": "AofF"}


def load_scriptures_json(path: Path, source: str, book_id: str | None = None) -> pd.DataFrame:
    """Load a book from the bcbooks/scriptures-json layout (public domain).

    Used privately for the Doctrine and Covenants (a ``sections`` layout; pass
    ``book_id="DC"``, giving ids like ``DC.27.15``) and the Pearl of Great Price (a
    ``books`` layout; ids from PEARL_BOOKS, like ``Moses.1.39``). Headings and
    editorial material are not in the files.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sections" in data:
        units = [(book_id, s["section"], s["verses"]) for s in data["sections"]]
    else:
        units = [(PEARL_BOOKS[b["lds_slug"]], c["chapter"], c["verses"])
                 for b in data["books"] for c in b["chapters"]]
    rows = [
        {"id": f"{bid}.{ch}.{v['verse']}", "title": v["reference"], "text": v["text"],
         "source": source, "book": bid, "chapter": int(ch), "verse": int(v["verse"])}
        for bid, ch, verses in units for v in verses
    ]
    return pd.DataFrame(rows)
