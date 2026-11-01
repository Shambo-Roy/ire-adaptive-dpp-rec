"""Greedy DPP and recommenders: hand examples, reference algorithms, kernel validity, list validity."""
import numpy as np
import pytest

from adaptdiv import seeding
from adaptdiv.catalogue import generate_catalogue
from adaptdiv.config import CatalogueConfig
from adaptdiv.dpp import brute_force_map, dpp_kernel, greedy_map, naive_greedy_map, objective
from adaptdiv.recommenders import (
    SCORE_EPS,
    log_scores,
    popularity_list,
    recommendation_lists,
    select,
    standard_methods,
)
from adaptdiv.relevance import top_k
from adaptdiv.similarity import min_eigenvalue, similarity_matrix

# Two near-duplicates with high scores, and one different item with a low score.
S3 = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
LOG3 = np.log([0.5, 0.4, 0.1])


def _instance(seed, n=14):
    rng = np.random.default_rng(seed)
    phi = generate_catalogue(CatalogueConfig(n_categories=3, items_per_category=10), seeding.stream(seed, seeding.CATALOGUE)).embeddings
    idx = rng.choice(30, n, replace=False)
    return np.log(rng.dirichlet(np.ones(n))), similarity_matrix(phi[idx])


def test_hand_example_diversity_beats_near_duplicate():
    """theta=0.5, item 1 vs 2: .5*log(.4)+.5*log(1-.81) = -1.288 < .5*log(.1) = -1.151."""
    assert list(greedy_map(LOG3, S3, 0.5, 2)) == [0, 2]


def test_hand_example_theta_one_ignores_similarity():
    assert list(greedy_map(LOG3, S3, 1.0, 2)) == [0, 1]


def test_theta_zero_ignores_scores():
    assert 2 in greedy_map(np.log([0.9, 0.09, 0.01]), S3, 0.0, 2)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("theta", [0.3, 0.5, 0.8])
def test_fast_greedy_equals_naive_greedy(seed, theta):
    log_s, S = _instance(seed)
    assert np.array_equal(greedy_map(log_s, S, theta, 6), naive_greedy_map(log_s, S, theta, 6))


@pytest.mark.parametrize("seed", range(6))
def test_greedy_never_beats_exact_and_stays_close(seed):
    log_s, S = _instance(seed)
    exact = objective(brute_force_map(log_s, S, 0.6, 5), log_s, S, 0.6)
    greedy = objective(greedy_map(log_s, S, 0.6, 5), log_s, S, 0.6)
    assert greedy <= exact + 1e-9
    assert exact - greedy < 1.0


def test_theta_one_equals_top_k():
    rng = np.random.default_rng(1)
    log_s = np.log(rng.dirichlet(np.ones(30)))
    S = similarity_matrix(generate_catalogue(CatalogueConfig(n_categories=3, items_per_category=10), rng).embeddings)
    assert set(greedy_map(log_s, S, 1.0, 7)) == set(top_k(log_s, 7))
    assert np.array_equal(select(log_s, S, 1.0, 7), top_k(log_s, 7))


def test_objective_equals_scaled_kernel_log_det():
    """F(Y) = (1 - theta) * log det L_Y: the objective really is the stated DPP."""
    log_s, S = _instance(2)
    theta, Y = 0.6, np.array([0, 3, 5, 9])
    L = dpp_kernel(np.exp(log_s), S, theta)
    assert np.isclose(objective(Y, log_s, S, theta), (1 - theta) * np.linalg.slogdet(L[np.ix_(Y, Y)])[1])


def test_kernel_is_symmetric_positive_definite():
    log_s, S = _instance(4)
    L = dpp_kernel(np.exp(log_s), S, 0.5)
    assert np.allclose(L, L.T)
    assert min_eigenvalue(L) > 0


def test_lists_are_distinct_even_with_duplicate_items():
    """Exact duplicates give d^2 = jitter-level gain; the floor must keep the selection valid."""
    phi = np.tile(np.eye(4), (5, 1))  # 20 items, only 4 distinct directions
    S = similarity_matrix(phi)
    Y = greedy_map(np.zeros(20), S, 0.5, 8)  # k exceeds the rank
    assert len(set(Y)) == 8
    assert len(set(np.argmax(phi[Y[:4]], axis=1))) == 4  # the first four cover all directions


def test_invalid_arguments():
    with pytest.raises(ValueError):
        greedy_map(LOG3, S3, 1.5, 2)
    with pytest.raises(ValueError):
        greedy_map(LOG3, S3, 0.5, 4)
    with pytest.raises(ValueError):
        dpp_kernel(np.ones(3), S3, 1.0)


# ------------------------------------------------------------------ recommenders
def _setup():
    cat = generate_catalogue(CatalogueConfig(n_categories=4, items_per_category=10), seeding.stream(0, seeding.CATALOGUE))
    r = np.full(40, 0.1)
    return cat.public(), r, similarity_matrix(cat.embeddings)


def test_log_scores_formula_and_zero_interest():
    pub, r, _ = _setup()
    z = np.array([1.0, 0.0, 0.0, 0.0])
    ls = log_scores(z, r, pub.categories)
    assert np.isfinite(ls).all()
    assert np.isclose(ls[0], np.log(0.1 * (1 + SCORE_EPS)))
    assert np.isclose(ls[39], np.log(0.1 * SCORE_EPS))


def test_recommendation_lists_shape_validity_and_steps():
    pub, r, S = _setup()
    prof = np.random.default_rng(0).dirichlet(np.ones(4), 12)
    all_lists = recommendation_lists(prof, r, pub.categories, S, 0.6, 5)
    assert all_lists.shape == (12, 5)
    assert all(len(set(row)) == 5 for row in all_lists)
    sub = recommendation_lists(prof, r, pub.categories, S, 0.6, 5, steps=np.array([3, 7]))
    assert np.array_equal(sub, all_lists[[3, 7]])


def test_list_uses_only_its_own_profile_row():
    """No look-ahead: changing later profile rows leaves earlier lists unchanged."""
    pub, r, S = _setup()
    prof = np.random.default_rng(1).dirichlet(np.ones(4), 10)
    changed = prof.copy()
    changed[6:] = np.random.default_rng(2).dirichlet(np.ones(4), 4)
    a = recommendation_lists(prof, r, pub.categories, S, 0.6, 5)
    b = recommendation_lists(changed, r, pub.categories, S, 0.6, 5)
    assert np.array_equal(a[:6], b[:6])


def test_dpp_covers_more_categories_than_top_k_when_interest_is_spread():
    pub, _, S = _setup()
    r = np.random.default_rng(3).dirichlet(np.ones(10), 4).reshape(-1)
    z = np.full((1, 4), 0.25)
    topk = recommendation_lists(z, r, pub.categories, S, 1.0, 6)[0]
    dpp = recommendation_lists(z, r, pub.categories, S, 0.3, 6)[0]
    assert len(set(pub.categories[dpp])) >= len(set(pub.categories[topk]))


def test_standard_methods_match_d16():
    methods = {m.name: m for m in standard_methods(0.6)}
    assert set(methods) == {"popularity", "static_topk", "adaptive_topk", "static_dpp", "adaptive_dpp", "oracle_topk", "oracle_dpp"}
    assert methods["adaptive_dpp"].theta == 0.6 and methods["adaptive_topk"].theta == 1.0


def test_popularity_list_is_fixed_top_k():
    assert list(popularity_list(np.array([0.1, 0.5, 0.4]), 2)) == [1, 2]
