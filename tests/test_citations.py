from rtm.citations import normalize, parse, parse_all

ORDER = (
    [f"Exod.17.{v}" for v in range(1, 17)]
    + [f"DC.27.{v}" for v in range(1, 19)]
    + [f"Alma.32.{v}" for v in range(40, 44)] + [f"Alma.33.{v}" for v in range(1, 4)]
    + [f"Ps.23.{v}" for v in range(1, 7)]
    + [f"2Ne.31.{v}" for v in range(1, 22)]
)


def test_normalize_spaces_and_garbled_dashes():
    assert normalize("Ex.\xa017:9�13") == "Ex. 17:9-13"
    assert normalize("JS—H 1:17.") == "JS-H 1:17"


def test_abbreviation_with_range():
    assert parse("Ex. 17:9–13", ORDER).verses == [f"Exod.17.{v}" for v in range(9, 14)]


def test_comma_list_continues_the_chapter():
    p = parse("D&C 27:15, 17", ORDER)
    assert p.verses == ["DC.27.15", "DC.27.17"]


def test_full_name_and_alias():
    assert parse("Doctrine and Covenants 27:1", ORDER).verses == ["DC.27.1"]
    assert parse("Psalm 23:1–3", ORDER).verses == ["Ps.23.1", "Ps.23.2", "Ps.23.3"]


def test_range_across_chapters():
    p = parse("Alma 32:42–33:2", ORDER)
    assert p.verses == ["Alma.32.42", "Alma.32.43", "Alma.33.1", "Alma.33.2"]


def test_chapter_citation_kept_separate():
    p = parse("2\xa0Ne. 31", ORDER)
    assert p.chapters == ["2Ne.31"] and p.verses == []


def test_prose_and_bare_verse_are_unparsed():
    assert parse("eighth chapter of Moroni, the twenty-fifth verse", ORDER).unparsed
    assert parse("verse 4", ORDER).unparsed
    assert parse("introduction to the Book of Mormon", ORDER).unparsed


def test_unknown_verse_ids_are_dropped():
    assert parse("Ex. 17:99", ORDER).verses == []


def test_parse_all_deduplicates():
    p = parse_all(["Ex. 17:9", "Exodus 17:9–10", "2 Ne. 31"], ORDER)
    assert p.verses == ["Exod.17.9", "Exod.17.10"] and p.chapters == ["2Ne.31"]


def test_bare_reference_continues_previous_book():
    p = parse_all(["D&C 27:1", "15–16", "Ex. 17:2", "17:4"], ORDER)
    # "15-16" alone is ambiguous (verses? chapters?) and is left unparsed; only
    # "c:v" forms inherit the book.
    assert "DC.27.1" in p.verses and "Exod.17.4" in p.verses


def test_section_and_chapter_words():
    assert parse("section 27:3", ORDER).verses == ["DC.27.3"]
    assert parse("Doctrine and Covenants section 27:4", ORDER).verses == ["DC.27.4"]
    assert parse("2 Nephi chapter 31", ORDER).chapters == ["2Ne.31"]
