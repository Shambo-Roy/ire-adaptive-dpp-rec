"""Interest estimators: hand-checked values, invariants, no look-ahead, edge cases."""
import numpy as np
import pytest

from adaptdiv.estimators import (
    EstimatorSpec,
    categories_of,
    cumulative_mean,
    ewma,
    ewma_mean_age,
    matched_window,
    sliding_window,
)

C = 3
CATS = np.array([0, 0, 1])


def test_ewma_hand_example():
    """alpha=0.5 from uniform: (1/3,1/3,1/3) -> (2/3,1/6,1/6) -> (5/6,1/12,1/12) -> (5/12,13/24,1/24)."""
    z = ewma(CATS, C, 0.5)
    expected = np.array([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [5 / 6, 1 / 12, 1 / 12], [5 / 12, 13 / 24, 1 / 24]])
    assert np.allclose(z, expected)


def test_ewma_matches_unrolled_formula():
    rng = np.random.default_rng(0)
    cats, alpha = rng.integers(0, 5, 40), 0.15
    z = ewma(cats, 5, alpha)
    x = np.eye(5)[cats]
    t = len(cats)
    weights = alpha * (1 - alpha) ** (t - 1 - np.arange(t))
    unrolled = (1 - alpha) ** t * np.full(5, 0.2) + weights @ x
    assert np.allclose(z[-1], unrolled)


def test_cumulative_mean_hand_example():
    z = cumulative_mean(CATS, C)
    expected = np.array([[1 / 3, 1 / 3, 1 / 3], [1, 0, 0], [1, 0, 0], [2 / 3, 1 / 3, 0]])
    assert np.allclose(z, expected)


def test_sliding_window_hand_example():
    z = sliding_window(np.array([0, 0, 1, 2]), C, window=2)
    expected = np.array([[1 / 3, 1 / 3, 1 / 3], [1, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5]])
    assert np.allclose(z, expected)


def test_limiting_cases():
    rng = np.random.default_rng(1)
    cats = rng.integers(0, 4, 30)
    last = np.eye(4)[cats]
    assert np.allclose(ewma(cats, 4, 1.0)[1:], last)               # alpha = 1: last observation
    assert np.allclose(sliding_window(cats, 4, 1)[1:], last)       # W = 1: last observation
    assert np.allclose(sliding_window(cats, 4, 1000), cumulative_mean(cats, 4))  # W >= T: cumulative mean


@pytest.mark.parametrize(
    "spec", [EstimatorSpec("cumulative"), EstimatorSpec("ewma", 0.1), EstimatorSpec("ewma", 0.7), EstimatorSpec("window", 7)]
)
def test_estimates_stay_on_simplex(spec):
    cats = np.random.default_rng(2).integers(0, 10, (6, 200))
    z = spec(cats, 10)
    assert z.shape == (6, 201, 10)
    assert (z >= -1e-12).all() and np.allclose(z.sum(axis=-1), 1.0)
    assert np.allclose(z[:, 0], 0.1)  # empty history -> uniform


@pytest.mark.parametrize("spec", [EstimatorSpec("cumulative"), EstimatorSpec("ewma", 0.2), EstimatorSpec("window", 5)])
def test_no_look_ahead(spec):
    """Changing observations after step t must not change estimates up to t."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 6, 100)
    b = a.copy()
    b[60:] = rng.integers(0, 6, 40)
    za, zb = spec(a, 6), spec(b, 6)
    assert np.allclose(za[:61], zb[:61])


def test_batch_equals_single():
    cats = np.random.default_rng(4).integers(0, 5, (3, 50))
    for spec in (EstimatorSpec("cumulative"), EstimatorSpec("ewma", 0.3), EstimatorSpec("window", 4)):
        batch = spec(cats, 5)
        for i in range(3):
            assert np.allclose(batch[i], spec(cats[i], 5))


def test_ewma_converges_to_observation_distribution():
    """Stationary input: the long-run average of EWMA equals the category distribution."""
    rng = np.random.default_rng(5)
    p = np.array([0.6, 0.3, 0.1])
    cats = rng.choice(3, size=50_000, p=p)
    z = ewma(cats, 3, 0.05)
    assert np.allclose(z[1000:].mean(axis=0), p, atol=0.01)


def test_isolated_noisy_observation_effect_is_alpha():
    """After a long run of category 0, one off-category observation moves EWMA by exactly alpha."""
    cats = np.array([0] * 200 + [2])
    z = ewma(cats, 3, 0.1)
    assert np.isclose(z[-1][2] - z[-2][2], 0.1 * (1 - z[-2][2]))
    assert z[-1].argmax() == 0  # a single noisy click does not change the dominant category


def test_matched_memory():
    assert np.isclose(ewma_mean_age(0.1), 9.0)
    assert matched_window(0.1) == 19          # (19 - 1) / 2 = 9 = mean age of EWMA(0.1)
    assert matched_window(1.0) == 1


def test_categories_of_maps_items():
    item_cats = np.repeat(np.arange(4), 3)
    assert np.array_equal(categories_of(np.array([0, 5, 11]), item_cats), [0, 1, 3])


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ewma(CATS, C, 0.0)
    with pytest.raises(ValueError):
        sliding_window(CATS, C, 0)
    with pytest.raises(ValueError):
        cumulative_mean(np.array([0, 3]), C)
