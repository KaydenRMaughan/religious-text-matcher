import numpy as np
import pandas as pd
import pytest

from rtm import search
from rtm.search import Index

DOCS = pd.DataFrame({
    "id": ["Isa.2.2", "2Ne.12.2", "Gen.1.1", "John.1.1"],
    "text": ["mountain of the lord's house established in the last days",
             "mountain of the lord's house established in the last days",
             "in the beginning god created the heaven and the earth",
             "in the beginning was the word"],
    "source": ["Old Testament", "Book of Mormon", "Old Testament", "New Testament"],
})

# Fake embeddings: a 3-word bag of "mountain", "beginning", "word".
VOCAB = ["mountain", "beginning", "word"]


def fake_vectors(texts, *_):
    return np.array([[t.split().count(w) for w in VOCAB] for t in texts], dtype=float) + 1e-9


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(search, "encode", lambda texts, key: fake_vectors(texts))
    monkeypatch.setattr(search, "unit_vectors", lambda texts, sources, key: fake_vectors(texts))


def test_bm25_search_ranks_and_ties_in_corpus_order():
    out = Index(DOCS).search("mountain of the lord", method="bm25", k=2)
    assert out["id"].tolist() == ["Isa.2.2", "2Ne.12.2"]
    assert out["rank"].tolist() == [1, 2]


def test_boost_breaks_the_tie_for_book_of_mormon():
    out = Index(DOCS).search("mountain of the lord", method="bm25", k=2, delta=1e-6)
    assert out["id"].tolist() == ["2Ne.12.2", "Isa.2.2"]


def test_source_filter():
    out = Index(DOCS).search("beginning", method="bm25", k=5, sources=["New Testament"])
    assert out["id"].tolist() == ["John.1.1"]


def test_dense_and_hybrid_run():
    idx = Index(DOCS)
    assert idx.search("the beginning", method="mpnet", k=1)["id"].iloc[0] in {"Gen.1.1", "John.1.1"}
    assert len(idx.search("the word", method="hybrid", k=3)) == 3
    with pytest.raises(ValueError):
        idx.scores("x", method="tfidf")


def test_chunked_documents_score_by_best_passage():
    long = pd.DataFrame({
        "id": ["talk-a", "talk-b"],
        "text": [" ".join(["Filler sentence here."] * 120) + " The mountain rises.",
                 " ".join(["Another plain sentence."] * 120)],
        "source": ["General Conference"] * 2,
    })
    idx = Index(long, chunk_docs=True)
    assert len(idx.passages[0]) > 1
    assert idx.search("mountain", method="minilm", k=1)["id"].iloc[0] == "talk-a"
