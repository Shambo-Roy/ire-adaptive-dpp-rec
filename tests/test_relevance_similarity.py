"""Base relevance, plug-in scores and the similarity kernel."""
import numpy as np

from adaptdiv import seeding
from adaptdiv.catalogue import generate_catalogue
from adaptdiv.config import CatalogueConfig
from adaptdiv.relevance import global_popularity, plugin_scores, top_k, within_category_popularity
from adaptdiv.similarity import DEFAULT_JITTER, min_eigenvalue, similarity_matrix
from adaptdiv.simulator import smoothed_popularity

CATS = np.array([0, 0, 1, 1])
COUNTS = np.array([3, 1, 0, 4])


def test_global_popularity_hand_example_matches_simulator():
    expected = np.array([4, 2, 1, 5]) / 12
    assert np.allclose(global_popularity(COUNTS), expected)
    assert np.allclose(global_popularity(COUNTS), smoothed_popularity(COUNTS))


def test_within_category_popularity_hand_example():
    assert np.allclose(within_category_popularity(COUNTS, CATS, 2), [4 / 6, 2 / 6, 1 / 6, 5 / 6])


def test_within_category_sums_to_one_and_positive():
    counts = np.random.default_rng(0).integers(0, 20, 100)
    cats = np.repeat(np.arange(10), 10)
    r = within_category_popularity(counts, cats, 10)
    assert (r > 0).all()
    assert np.allclose(r.reshape(10, 10).sum(axis=1), 1.0)


def test_empty_category_is_uniform():
    r = within_category_popularity(np.zeros(4, dtype=int), CATS, 2)
    assert np.allclose(r, 0.5)


def test_plugin_scores_hand_example():
    r = within_category_popularity(COUNTS, CATS, 2)
    s = plugin_scores(np.array([0.25, 0.75]), r, CATS)
    assert np.allclose(s, [0.25 * 4 / 6, 0.25 * 2 / 6, 0.75 / 6, 0.75 * 5 / 6])


def test_plugin_scores_are_a_distribution_and_batched():
    rng = np.random.default_rng(1)
    cats = np.repeat(np.arange(5), 4)
    r = within_category_popularity(rng.integers(0, 9, 20), cats, 5)
    zhat = rng.dirichlet(np.ones(5), size=(3, 7))
    s = plugin_scores(zhat, r, cats)
    assert s.shape == (3, 7, 20)
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert np.allclose(s[2, 3], plugin_scores(zhat[2, 3], r, cats))


def test_zero_interest_category_scores_zero():
    """A zero estimate zeroes the score. The DPP quality transform must handle this (epsilon, D-14)."""
    r = within_category_popularity(COUNTS, CATS, 2)
    assert np.allclose(plugin_scores(np.array([1.0, 0.0]), r, CATS)[2:], 0.0)


def test_top_k_order_ties_and_batch():
    assert list(top_k(np.array([0.1, 0.4, 0.4, 0.2]), 3)) == [1, 2, 3]
    batch = top_k(np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]]), 2)
    assert batch.tolist() == [[0, 2], [1, 2]]


def _embeddings(seed=0):
    return generate_catalogue(CatalogueConfig(), seeding.stream(seed, seeding.CATALOGUE)).embeddings


def test_similarity_is_symmetric_with_unit_diagonal():
    S = similarity_matrix(_embeddings())
    assert S.shape == (500, 500)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0 + DEFAULT_JITTER)
    assert S.max() <= 1.0 + DEFAULT_JITTER + 1e-12 and S.min() >= -1.0 - 1e-12


def test_similarity_is_positive_definite_with_jitter():
    assert min_eigenvalue(similarity_matrix(_embeddings())) > 0.5 * DEFAULT_JITTER


def test_gram_matrix_psd_with_rank_at_most_d():
    S0 = similarity_matrix(_embeddings(), jitter=0.0)
    assert min_eigenvalue(S0) > -1e-9
    assert np.linalg.matrix_rank(S0, tol=1e-8) == 32


def test_similarity_does_not_modify_embeddings():
    phi = _embeddings()
    before = phi.copy()
    similarity_matrix(phi)
    assert np.array_equal(phi, before)


def test_duplicate_items_have_zero_volume():
    """Near-duplicates give a near-zero determinant: the property that drives DPP diversity."""
    phi = _embeddings()
    distinct = similarity_matrix(phi[[0, 100]], jitter=0.0)
    duplicate = similarity_matrix(phi[[0, 0]], jitter=0.0)
    assert np.linalg.det(distinct) > 0.5
    assert abs(np.linalg.det(duplicate)) < 1e-12
