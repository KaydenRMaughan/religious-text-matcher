import math

import numpy as np
import pytest
import snowballstemmer
from rank_bm25 import BM25Okapi

from rtm.retrieve import BM25, tokenize

SNOWBALL = snowballstemmer.stemmer("english")


# --- tokenizer --------------------------------------------------------------------

def test_lowercases_and_splits_on_punctuation():
    assert tokenize("And God said, Let there be light: and there was light.") == [
        "and", "god", "said", "let", "there", "be", "light", "and", "there", "was", "light",
    ]


def test_curly_and_straight_apostrophes_match():
    assert tokenize("my father’s house") == tokenize("my father's house") == [
        "my", "father's", "house",
    ]


def test_hyphen_separates():
    assert tokenize("rich toward God-ward") == ["rich", "toward", "god", "ward"]


def test_no_stopword_removal():
    assert tokenize("thee and thou") == ["thee", "and", "thou"]


def test_stemming_is_off_by_default():
    assert tokenize("loveth") == ["loveth"]


def test_stemming_collapses_modern_inflections():
    assert {tokenize(w, stem=True)[0] for w in ["love", "loved", "loving", "loves"]} == {"love"}


@pytest.mark.parametrize(
    "archaic, modern",
    [
        ("loveth", "loves"), ("lovest", "loves"), ("cometh", "comes"), ("knoweth", "knows"),
        ("knowest", "knows"), ("believeth", "believes"), ("dwelleth", "dwells"),
        ("giveth", "gives"), ("speaketh", "speaks"), ("goeth", "goes"),
    ],
)
def test_stemming_maps_kjv_verb_endings(archaic, modern):
    assert tokenize(archaic, stem=True) == tokenize(modern, stem=True)


@pytest.mark.parametrize(
    "word",
    ["priest", "rest", "lest", "west", "manifest", "harvest", "forest", "beast",
     "breast", "honest", "seth", "teeth", "nazareth"],
)
def test_stemming_leaves_non_verb_eth_est_words_to_snowball(word):
    assert tokenize(word, stem=True) == [SNOWBALL.stemWord(word)]


# --- BM25 -----------------------------------------------------------------------------

DOCS = [
    "and the lord spake unto moses saying".split(),
    "and moses went up unto the mount".split(),
    "the lord is my shepherd i shall not want".split(),
    "and it came to pass".split(),
    "love the lord thy god with all thy heart and all thy soul".split(),
]


class LuceneOkapi(BM25Okapi):
    """rank-bm25's scorer with Lucene's IDF swapped in: the reference for BM25."""

    def _calc_idf(self, nd):
        for word, n in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5))


@pytest.mark.parametrize("k1, b", [(1.5, 0.75), (1.2, 0.75), (2.0, 0.3), (0.9, 1.0)])
@pytest.mark.parametrize(
    "query",
    [["lord"], ["and", "lord"], ["thy", "thy"], ["moses", "mount", "unknownword"], ["the"]],
)
def test_scores_match_reference(k1, b, query):
    expected = LuceneOkapi(DOCS, k1=k1, b=b).get_scores(query)
    np.testing.assert_allclose(BM25(DOCS, k1=k1, b=b).scores(query), expected, rtol=1e-9)


def test_scores_shape_and_order():
    s = BM25(DOCS).scores(["shepherd"])
    assert s.shape == (len(DOCS),)
    assert np.argmax(s) == 2


def test_idf_stays_positive_for_common_terms():
    bm = BM25(DOCS)
    assert bm.idf("the") > 0  # in 4 of 5 docs: Okapi's IDF would be negative here
    assert bm.idf("shepherd") > bm.idf("lord") > bm.idf("the")


def test_unknown_terms_and_empty_query_score_zero():
    bm = BM25(DOCS)
    assert bm.idf("unknownword") == 0.0
    np.testing.assert_array_equal(bm.scores(["unknownword"]), np.zeros(len(DOCS)))
    np.testing.assert_array_equal(bm.scores([]), np.zeros(len(DOCS)))


def test_b_zero_ignores_document_length():
    docs = [["grace"], ["grace", "x", "y", "z", "w"]]
    s = BM25(docs, b=0.0).scores(["grace"])
    assert s[0] == pytest.approx(s[1])


def test_k1_zero_makes_term_frequency_binary():
    docs = [["grace"], ["grace", "grace", "grace"], ["x"]]
    s = BM25(docs, k1=0.0).scores(["grace"])
    assert s[0] == pytest.approx(s[1])
    assert s[2] == 0.0  # 0/0 when f = 0 and k1 = 0: must be 0, not NaN


def test_term_frequency_saturates():
    docs = [["grace"] * n + ["x"] * (10 - n) for n in (1, 2, 3, 4)] + [["y"] * 10]
    s = BM25(docs).scores(["grace"])[:4]
    gains = np.diff(s)
    assert np.all(gains > 0) and np.all(np.diff(gains) < 0)


# --- dense retrieval -------------------------------------------------------------------

from rtm.retrieve import DenseRetriever, l2_normalize  # noqa: E402

VECS = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])


def test_l2_normalize_rows_have_unit_length():
    out = l2_normalize(VECS)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)
    np.testing.assert_allclose(out[0], [0.6, 0.8])


def test_l2_normalize_one_vector_and_zero_vector():
    np.testing.assert_allclose(l2_normalize(np.array([0.0, 5.0])), [0.0, 1.0])
    out = l2_normalize(np.array([[0.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_array_equal(out, [[0.0, 0.0], [1.0, 0.0]])
    assert not np.isnan(out).any()


def test_l2_normalize_leaves_input_alone_and_returns_float64():
    x = np.array([[3, 4]], dtype=np.float32)
    out = l2_normalize(x)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(x, [[3, 4]])


def test_dense_scores_are_cosines():
    q = np.array([1.0, 1.0])
    expected = VECS @ q / (np.linalg.norm(VECS, axis=1) * np.linalg.norm(q))
    s = DenseRetriever(VECS).scores(q)
    assert s.shape == (4,)
    np.testing.assert_allclose(s, expected)


def test_dense_scores_ignore_query_length_and_bound():
    r = DenseRetriever(VECS)
    q = np.array([2.0, -1.0])
    np.testing.assert_allclose(r.scores(q), r.scores(10 * q))
    s = r.scores(q)
    assert np.all(s <= 1 + 1e-12) and np.all(s >= -1 - 1e-12)


def test_dense_identical_opposite_orthogonal():
    s = DenseRetriever(VECS).scores(np.array([1.0, 0.0]))
    assert s[1] == pytest.approx(1.0)       # same direction
    assert s[2] == pytest.approx(0.0)       # orthogonal
    assert s[3] == pytest.approx(-1 / np.sqrt(2))


# --- reciprocal rank fusion ------------------------------------------------------------

from rtm.retrieve import rrf_scores  # noqa: E402


def test_rrf_hand_computed():
    a = np.array([3.0, 2.0, 1.0])     # ranks 1, 2, 3
    b = np.array([0.1, 0.9, 0.5])     # ranks 3, 1, 2
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    np.testing.assert_allclose(rrf_scores([a, b]), expected)


def test_rrf_uses_ranks_not_scale():
    a = np.array([3.0, 2.0, 1.0]); b = np.array([0.1, 0.9, 0.5])
    np.testing.assert_allclose(rrf_scores([a, b]), rrf_scores([1000 * a, b - 7]))


def test_rrf_single_retriever_keeps_its_order():
    a = np.array([0.2, 0.9, 0.5, 0.1])
    assert np.argsort(-rrf_scores([a]), kind="stable").tolist() == [1, 2, 0, 3]


def test_rrf_ties_break_by_corpus_order():
    s = rrf_scores([np.array([1.0, 1.0, 0.0])], c=0)
    np.testing.assert_allclose(s, [1.0, 0.5, 1 / 3])


def test_rrf_rejects_bad_input():
    with pytest.raises(ValueError):
        rrf_scores([])
    with pytest.raises(ValueError):
        rrf_scores([np.zeros(3), np.zeros(4)])
