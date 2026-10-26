"""Latent interests, half-swap shift and drift trajectories."""
import numpy as np
import pytest

from adaptdiv.config import SCENARIOS, ScenarioConfig
from adaptdiv.scenarios import (
    apply_permutation,
    half_swap_permutation,
    mixing_weights,
    noise_rate,
    sample_base_interest,
    shifted_interest,
    trajectory,
)

C = 10
CFG = ScenarioConfig()


def tv(a, b):
    return 0.5 * np.abs(a - b).sum(axis=-1)


def _pair(seed=0, kappa=0.3):
    rng = np.random.default_rng(seed)
    z_a = sample_base_interest(kappa, C, rng)
    perm = half_swap_permutation(z_a, rng)
    return z_a, shifted_interest(z_a, perm), perm


@pytest.mark.parametrize("kappa", [0.1, 1.0])
def test_base_interest_on_simplex(kappa):
    rng = np.random.default_rng(1)
    for _ in range(500):
        z = sample_base_interest(kappa, C, rng)
        assert (z >= 0).all() and np.isclose(z.sum(), 1.0)


@pytest.mark.parametrize("kappa,expected_max", [(0.1, 0.665), (1.0, 0.293)])
def test_concentration_matches_design_evidence(kappa, expected_max):
    rng = np.random.default_rng(2)
    zmax = np.mean([sample_base_interest(kappa, C, rng).max() for _ in range(5000)])
    assert abs(zmax - expected_max) < 0.02


def test_half_swap_is_permutation_swapping_halves():
    for seed in range(50):
        z_a, z_b, perm = _pair(seed)
        assert sorted(perm) == list(range(C))
        top = set(np.argsort(-z_a, kind="stable")[: C // 2])
        assert all(perm[c] not in top for c in top)  # every top category moves into the bottom half
        assert np.allclose(np.sort(z_a), np.sort(z_b))  # same concentration
        assert np.isclose(z_b.sum(), 1.0)


@pytest.mark.parametrize("kappa,expected_tv", [(0.1, 0.99), (1.0, 0.65)])
def test_shift_size_matches_design_evidence(kappa, expected_tv):
    tvs = [tv(*_pair(s, kappa)[:2]) for s in range(3000)]
    assert abs(np.mean(tvs) - expected_tv) < 0.03


def test_partial_shift_magnitude():
    z_a, z_full, perm = _pair(5)
    z_half = shifted_interest(z_a, perm, 0.5)
    assert np.isclose(tv(z_a, z_half), 0.5 * tv(z_a, z_full))
    assert np.allclose(shifted_interest(z_a, perm, 0.0), z_a)
    assert np.allclose(apply_permutation(z_a, perm), z_full)


@pytest.mark.parametrize("name", SCENARIOS)
def test_trajectories_on_simplex(name):
    z_a, z_b, _ = _pair(3)
    z = trajectory(name, z_a, z_b, CFG)
    assert z.shape == (CFG.horizon, C)
    assert (z >= 0).all() and np.allclose(z.sum(axis=1), 1.0)


def test_switch_points():
    z_a, z_b, _ = _pair(4)
    t = np.arange(1, CFG.horizon + 1)

    assert np.allclose(trajectory("stable", z_a, z_b, CFG), z_a)
    assert np.allclose(trajectory("noisy", z_a, z_b, CFG), z_a)

    ab = trajectory("abrupt", z_a, z_b, CFG)
    assert np.allclose(ab[t < 100], z_a) and np.allclose(ab[t >= 100], z_b)

    rec = trajectory("recurrent", z_a, z_b, CFG)
    assert np.allclose(rec[t < 70], z_a)
    assert np.allclose(rec[(t >= 70) & (t < 140)], z_b)
    assert np.allclose(rec[t >= 140], z_a)

    w = mixing_weights("gradual", CFG)
    assert np.all(w[t <= 80] == 0) and np.all(w[t >= 120] == 1)
    assert np.all(np.diff(w) >= 0)
    assert np.isclose(w[t == 100][0], 0.5)


def test_noise_rates():
    assert noise_rate("noisy", 0.05, CFG) == 0.3
    for s in ("stable", "abrupt", "gradual", "recurrent"):
        assert noise_rate(s, 0.05, CFG) == 0.05
    with pytest.raises(ValueError):
        noise_rate("unknown", 0.05, CFG)
