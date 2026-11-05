import pytest

from rtm.chunk import chunk_text, split_sentences


def test_split_sentences():
    text = 'He said, "Come." Then they went. Was it 3 p.m.? Yes! 2 Nephi 2:25 says so.'
    assert split_sentences(text) == [
        'He said, "Come."', "Then they went.", "Was it 3 p.m.?", "Yes!", "2 Nephi 2:25 says so.",
    ]


def test_short_text_is_one_chunk_and_empty_is_none():
    assert chunk_text("One. Two.") == ["One. Two."]
    assert chunk_text("   ") == []


def test_windows_respect_limit_and_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(10))  # 6 words each
    chunks = chunk_text(text, max_words=20, overlap=1)
    assert all(len(c.split()) <= 20 for c in chunks)
    first, second = chunks[0].split(". "), chunks[1].split(". ")
    assert first[-1].rstrip(".") == second[0].rstrip(".")      # one sentence repeated
    joined = " ".join(chunks)
    assert all(f"number {i} " in joined for i in range(10))     # nothing lost


def test_long_sentence_is_split_on_words():
    chunks = chunk_text(" ".join(["word"] * 450) + ".", max_words=200, overlap=0)
    assert [len(c.split()) for c in chunks] == [200, 200, 50]


def test_no_overlap_partitions_the_text():
    text = " ".join(f"S{i} a b c." for i in range(12))
    chunks = chunk_text(text, max_words=8, overlap=0)
    assert " ".join(chunks) == text


def test_bad_arguments():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0)
