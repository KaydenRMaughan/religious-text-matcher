"""Scripture citation parsing (Claude's: engineering for the private conference test).

Turns citation strings such as ``Ex. 17:9–13``, ``D&C 27:15, 17`` or ``2 Ne. 31`` into
verse ids in the corpus id scheme (``Exod.17.9``, ``DC.27.15``, ``2Ne.31``). Generic:
it holds no conference text, only book names and abbreviations.

Two kinds of result, kept apart so the evaluation protocol can decide how to use them:
- verse citations (``book c:v``), expanded to individual verse ids;
- chapter citations (``book c``), returned as chapter ids (``2Ne.31``).

Strings that aren't a book-and-number citation (prose such as "the eighth chapter of
Moroni", or bare "verse 4", which needs the previous citation) are reported as
unparsed rather than guessed at.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from rtm.corpus import BIBLE_BOOKS, BOOK_OF_MORMON_BOOKS

# Church-style abbreviations and other spellings seen in citations, beyond each
# book's display name. Keys are matched after _key(): lowercase, no dots or spaces.
_EXTRA = {
    "Gen": "Gen", "Ex": "Exod", "Exod": "Exod", "Lev": "Lev", "Num": "Num",
    "Deut": "Deut", "Josh": "Josh", "Judg": "Judg", "1Sam": "1Sam", "2Sam": "2Sam",
    "1Kgs": "1Kgs", "2Kgs": "2Kgs", "1Chr": "1Chr", "2Chr": "2Chr", "Neh": "Neh",
    "Esth": "Esth", "Ps": "Ps", "Psalm": "Ps", "Prov": "Prov", "Eccl": "Eccl",
    "Song": "Song", "Isa": "Isa", "Jer": "Jer", "Lam": "Lam", "Ezek": "Ezek",
    "Dan": "Dan", "Obad": "Obad", "Hab": "Hab", "Zeph": "Zeph", "Hag": "Hag",
    "Zech": "Zech", "Mal": "Mal", "Matt": "Matt", "Rom": "Rom", "1Cor": "1Cor",
    "2Cor": "2Cor", "Gal": "Gal", "Eph": "Eph", "Philip": "Phil", "Phil": "Phil",
    "Col": "Col", "1Thes": "1Thess", "2Thes": "2Thess", "1Thess": "1Thess",
    "2Thess": "2Thess", "1Tim": "1Tim", "2Tim": "2Tim", "Philem": "Phlm",
    "Heb": "Heb", "1Pet": "1Pet", "2Pet": "2Pet", "1Jn": "1John", "2Jn": "2John",
    "3Jn": "3John", "Rev": "Rev",
    "1Ne": "1Ne", "2Ne": "2Ne", "WofM": "WofM", "Hel": "Hel", "3Ne": "3Ne",
    "4Ne": "4Ne", "Morm": "Morm", "Moro": "Moro",
    "D&C": "DC", "DoctrineandCovenants": "DC",
    "Moses": "Moses", "Abr": "Abr", "Abraham": "Abr", "JS-M": "JSM", "JS-H": "JSH",
    "JosephSmith-Matthew": "JSM", "JosephSmith-History": "JSH",
    "AofF": "AofF", "ArticlesofFaith": "AofF",
}

DASHES = "‐‑‒–—―�"  # �: dashes garbled in storage


def _key(name: str) -> str:
    return re.sub(r"[.\s]", "", name).lower()


ALIASES: dict[str, str] = {
    **{_key(name): osis for osis, name in BIBLE_BOOKS},
    **{_key(osis): osis for osis, _ in BIBLE_BOOKS},
    **{_key(name): bid for bid, name, _ in BOOK_OF_MORMON_BOOKS},
    **{_key(k): v for k, v in _EXTRA.items()},
}

_CITATION = re.compile(r"^(?P<book>(?:[1-4]\s*)?[A-Za-z&][A-Za-z&.\s-]*?)\s*(?P<rest>\d.*)$")
_PART = re.compile(r"^(\d+)(?::(\d+))?(?:-(?:(\d+):)?(\d+))?$")


@dataclass
class Parsed:
    verses: list[str] = field(default_factory=list)     # e.g. "Exod.17.9"
    chapters: list[str] = field(default_factory=list)   # e.g. "2Ne.31"
    unparsed: list[str] = field(default_factory=list)


def normalize(citation: str) -> str:
    """Non-breaking spaces to spaces; every dash variant (and U+FFFD) to '-'."""
    text = citation.replace("\xa0", " ")
    for dash in DASHES:
        text = text.replace(dash, "-")
    text = re.sub(r"\s*-\s*", "-", text)       # "JS - H" -> "JS-H", "9 - 13" -> "9-13"
    text = re.sub(r"^[Ss]ection\s+(?=\d)", "D&C ", text)            # "section 89"
    text = re.sub(r"\s+(?:chapter|section)\s+(?=\d)", " ", text)    # "Alma chapter 32"
    return re.sub(r"\s+", " ", text).strip().strip("()").rstrip(".;,")


def parse(citation: str, verse_order: Sequence[str], default_book: str | None = None) -> Parsed:
    """Parse one citation string against a corpus's verse ids (in canonical order).

    ``verse_order`` lets ranges expand across chapter boundaries (``Alma 32:41-33:2``)
    and drops ids the corpus doesn't have. ``default_book`` (a citation's book name,
    as written) fills in a bare ``36:16-19``, which in footnotes continues the
    previous citation's book.
    """
    out = Parsed()
    text = normalize(citation)
    if default_book and re.match(r"^\d+:", text):
        text = f"{default_book} {text}"
    m = _CITATION.match(text)
    book = ALIASES.get(_key(m["book"])) if m else None
    if book is None:
        out.unparsed.append(citation)
        return out
    known = _index(verse_order)
    chapter = None
    for part in (p.strip() for p in m["rest"].split(",")):
        pm = _PART.match(part.replace(" ", ""))
        if pm is None:
            out.unparsed.append(citation)
            return out
        a, b, c2, d = pm.groups()
        if b is None and d is None:           # "31": a chapter, or a verse after "c:v,"
            if chapter is None:
                out.chapters.append(f"{book}.{a}")
            else:
                out.verses += _expand(book, chapter, int(a), chapter, int(a), known, verse_order)
            continue
        if b is None:                           # "16-17": verses after "c:v," or chapters
            if chapter is None:
                out.chapters += [f"{book}.{n}" for n in range(int(a), int(d) + 1)]
            else:
                out.verses += _expand(book, chapter, int(a), chapter, int(d), known, verse_order)
            continue
        chapter = int(a)
        end_ch = int(c2) if c2 else chapter
        end_v = int(d) if d else int(b)
        out.verses += _expand(book, chapter, int(b), end_ch, end_v, known, verse_order)
        chapter = end_ch
    return out


def parse_all(citations: Iterable[str], verse_order: Sequence[str]) -> Parsed:
    """Parse one talk's citations, in order; ids deduplicated, order kept.

    A bare ``c:v`` takes the book of the most recent citation that named one.
    """
    total = Parsed()
    last_book = None
    for c in citations:
        p = parse(c, verse_order, default_book=last_book)
        m = _CITATION.match(normalize(c))
        if m and m["book"].strip() and _key(m["book"]) in ALIASES:
            last_book = m["book"]
        total.verses += p.verses
        total.chapters += p.chapters
        total.unparsed += p.unparsed
    total.verses = list(dict.fromkeys(total.verses))
    total.chapters = list(dict.fromkeys(total.chapters))
    return total


_INDEX_CACHE: dict[int, dict[str, int]] = {}


def _index(verse_order: Sequence[str]) -> dict[str, int]:
    key = id(verse_order)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = {v: i for i, v in enumerate(verse_order)}
    return _INDEX_CACHE[key]


def _expand(book, ch1, v1, ch2, v2, known, order) -> list[str]:
    start, end = known.get(f"{book}.{ch1}.{v1}"), known.get(f"{book}.{ch2}.{v2}")
    if start is None:
        return []
    if end is None or end < start:
        return [order[start]]
    return list(order[start:end + 1])
