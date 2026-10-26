"""Interaction generation, common random numbers, reproducibility and the leakage firewall."""
from dataclasses import replace

import numpy as np

from adaptdiv import seeding
from adaptdiv.catalogue import generate_catalogue
from adaptdiv.config import SCENARIOS, CatalogueConfig, InteractionConfig, PopulationConfig, SimulationConfig
from adaptdiv.simulator import (
    OBSERVED_FIELDS,
    generate,
    item_conditional,
    sample_interactions,
    smoothed_popularity,
)

C, N_C = 10, 50


def _catalogue():
    return generate_catalogue(CatalogueConfig(), seeding.stream(0, seeding.CATALOGUE))


def _taste(seed=0):
    u = np.random.default_rng(seed).standard_normal(32)
    return u / np.linalg.norm(u)


# ----------------------------------------------------------------- generative model
def test_item_conditional_rows_are_distributions():
    cond = item_conditional(_catalogue(), _taste(), InteractionConfig())
    assert cond.shape == (C, N_C)
    assert (cond > 0).all() and np.allclose(cond.sum(axis=1), 1.0)


def test_item_conditional_without_taste_is_popularity_power():
    cat = _catalogue()
    cond = item_conditional(cat, _taste(), InteractionConfig(gamma=2.0, taste_weight=0.0))
    expected = cat.popularity.reshape(C, N_C) ** 2
    assert np.allclose(cond, expected / expected.sum(axis=1, keepdims=True))


def test_empirical_frequencies_match_model():
    """Long stable sequence: category and item frequencies converge to the model."""
    T, eta = 200_000, 0.2
    rng = np.random.default_rng(11)
    z = rng.dirichlet(np.full(C, 0.5))
    cond = item_conditional(_catalogue(), _taste(), InteractionConfig())
    items, cats, noise = sample_interactions(np.broadcast_to(z, (T, C)), eta, cond, rng.random((3, T)))

    assert abs(noise.mean() - eta) < 0.005
    cat_freq = np.bincount(cats, minlength=C) / T
    assert 0.5 * np.abs(cat_freq - ((1 - eta) * z + eta / C)).sum() < 0.01
    assert np.array_equal(items // N_C, cats)

    expected = (((1 - eta) * z + eta / C)[:, None] * cond).reshape(-1)
    item_freq = np.bincount(items, minlength=C * N_C) / T
    assert 0.5 * np.abs(item_freq - expected).sum() < 0.03


def test_item_distribution_is_exact_marginal(small_data):
    user = small_data.users[(0.1, "noisy")][0]
    gt = user.truth
    for k in (0, 99, 199):
        p = gt.item_distribution(k)
        assert np.isclose(p.sum(), 1.0) and (p > 0).all()
        cat_mass = p.reshape(C, N_C).sum(axis=1)
        assert np.allclose(cat_mass, (1 - gt.noise_rate) * gt.z[k] + gt.noise_rate / C)


def test_items_consistent_with_categories(small_data):
    for (kappa, scenario), users in small_data.users.items():
        for u in users:
            items, gt = u.observed.items, u.truth
            assert items.min() >= 0 and items.max() < C * N_C
            assert np.array_equal(items // N_C, gt.item_categories)


# ----------------------------------------------------------------- common random numbers
def test_scenarios_identical_before_first_switch(small_data):
    for kappa in small_data.config.population.kappas:
        for j in range(5):
            stable = small_data.users[(kappa, "stable")][j].observed.items
            abrupt = small_data.users[(kappa, "abrupt")][j].observed.items
            recur = small_data.users[(kappa, "recurrent")][j].observed.items
            gradual = small_data.users[(kappa, "gradual")][j].observed.items
            assert np.array_equal(stable[:99], abrupt[:99])    # t < 100
            assert np.array_equal(stable[:69], recur[:69])     # t < 70
            assert np.array_equal(stable[:80], gradual[:80])   # w_t = 0 for t <= 80


def test_noisy_scenario_differs_only_where_noise_threshold_matters(small_data):
    for j in range(5):
        s = small_data.users[(1.0, "stable")][j].truth
        n = small_data.users[(1.0, "noisy")][j].truth
        assert np.all(n.noise_flags[s.noise_flags])  # eta 0.3 > 0.05: every stable noise flag is also noisy
        same_flags = s.noise_flags == n.noise_flags
        assert np.array_equal(
            small_data.users[(1.0, "stable")][j].observed.items[same_flags],
            small_data.users[(1.0, "noisy")][j].observed.items[same_flags],
        )


def test_drift_visible_in_interactions(small_data):
    """Abrupt: category frequencies after the switch follow z_B, not z_A (pooled over users)."""
    for kappa in small_data.config.population.kappas:
        users = small_data.users[(kappa, "abrupt")]
        tv_b, tv_a = [], []
        for u in users:
            post = np.bincount(u.truth.item_categories[100:], minlength=C) / 100
            target_b = 0.95 * u.truth.params.z_b + 0.05 / C
            target_a = 0.95 * u.truth.params.z_a + 0.05 / C
            tv_b.append(0.5 * np.abs(post - target_b).sum())
            tv_a.append(0.5 * np.abs(post - target_a).sum())
        assert np.mean(tv_b) < np.mean(tv_a) - 0.2


# ----------------------------------------------------------------- reproducibility
def test_generate_is_reproducible(small_cfg, small_data):
    again = generate(seed=0, cfg=small_cfg)
    assert np.array_equal(again.train_item_counts, small_data.train_item_counts)
    for key, users in small_data.users.items():
        for a, b in zip(users, again.users[key]):
            assert np.array_equal(a.observed.items, b.observed.items)


def test_adding_users_does_not_change_existing_users(small_cfg, small_data):
    bigger = replace(small_cfg, population=replace(small_cfg.population, users_per_cell=25))
    more = generate(seed=0, cfg=bigger, scenarios=("abrupt",))
    for j in range(20):
        assert np.array_equal(
            more.users[(0.1, "abrupt")][j].observed.items,
            small_data.users[(0.1, "abrupt")][j].observed.items,
        )


def test_different_seeds_differ(small_cfg, small_data):
    other = generate(seed=1, cfg=small_cfg, scenarios=("stable",))
    assert not np.array_equal(
        other.users[(0.1, "stable")][0].observed.items, small_data.users[(0.1, "stable")][0].observed.items
    )


def test_users_share_params_across_scenarios(small_data):
    for j in range(5):
        params = {s: small_data.users[(0.1, s)][j].truth.params for s in SCENARIOS}
        ref = params["stable"]
        assert all(np.array_equal(p.z_a, ref.z_a) and np.array_equal(p.taste, ref.taste) for p in params.values())


# ----------------------------------------------------------------- firewall
def test_observed_sequence_carries_only_item_ids(small_data):
    assert OBSERVED_FIELDS == {"items"}
    obs = small_data.users[(0.1, "abrupt")][0].observed
    assert obs.items.dtype.kind == "i"
    assert not obs.items.flags.writeable
    assert np.array_equal(obs.prefix(10), obs.items[:10])


def test_ground_truth_arrays_read_only(small_data):
    gt = small_data.users[(1.0, "gradual")][0].truth
    for a in (gt.z, gt.noise_flags, gt.item_conditional, gt.params.z_a, gt.params.taste):
        assert not a.flags.writeable


# ----------------------------------------------------------------- training population
def test_training_counts_and_smoothed_popularity(small_cfg, small_data):
    counts = small_data.train_item_counts
    pop = small_cfg.population
    assert counts.sum() == pop.train_users * pop.train_horizon
    r = smoothed_popularity(counts)
    assert (r > 0).all() and np.isclose(r.sum(), 1.0)
    assert np.isclose(r[0], (counts[0] + 1) / (counts.sum() + counts.shape[0]))


def test_training_population_independent_of_evaluation_users(small_cfg):
    a = generate(seed=0, cfg=small_cfg, scenarios=("stable",))
    fewer = replace(small_cfg, population=PopulationConfig(users_per_cell=3, train_users=100))
    b = generate(seed=0, cfg=fewer, scenarios=("stable",))
    assert np.array_equal(a.train_item_counts, b.train_item_counts)
