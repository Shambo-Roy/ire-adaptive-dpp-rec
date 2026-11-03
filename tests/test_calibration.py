"""Calibrated re-ranker (D-19): hand-checked KL, Top-K limit, greedy vs direct objective, proportions."""
import numpy as np
import pytest

from adaptdiv.calibration import calibrated_lists, kl_to_smoothed, miscalibration
from adaptdiv.relevance import top_k


def test_kl_hand_example():
    """p = (.7, .3), list all category 0: q~ = (.997, .003); KL = .7 log(.7/.997) + .3 log(.3/.003)."""
    expected = 0.7 * np.log(0.7 / 0.997) + 0.3 * np.log(0.3 / 0.003)
    assert np.isclose(kl_to_smoothed(np.array([0.7, 0.3]), np.array([1.0, 0.0])), expected)


def test_kl_is_zero_for_a_calibrated_list_and_ignores_zero_targets():
    assert np.isclose(miscalibration(np.array([[0, 0, 1, 1]]), np.array([[0.5, 0.5, 0.0]]), 3)[0], 0.0)
    assert np.isfinite(miscalibration(np.array([[2, 2, 2, 2]]), np.array([[0.5, 0.5, 0.0]]), 3)[0])


def _instance(seed, n_rows=3, C=4, per_cat=5):
    rng = np.random.default_rng(seed)
    categories = np.repeat(np.arange(C), per_cat)
    log_s = np.log(rng.dirichlet(np.ones(C * per_cat), size=n_rows))
    targets = rng.dirichlet(np.full(C, 0.5), size=n_rows)
    return log_s, targets, categories


@pytest.mark.parametrize("seed", range(5))
def test_lambda_zero_is_top_k(seed):
    log_s, targets, categories = _instance(seed)
    assert np.array_equal(calibrated_lists(log_s, targets, categories, 0.0, 6), top_k(log_s, 6))


def _objective(items, log_s, target, categories, lam):
    q = np.bincount(categories[items], minlength=target.shape[0]) / len(items)
    return (1 - lam) * log_s[items].sum() - lam * kl_to_smoothed(target, q)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("lam", [0.3, 0.8, 0.99])
def test_greedy_matches_a_direct_objective_greedy(seed, lam):
    """Each step adds the item that maximises the full objective of the larger list (paper, Eq. 6)."""
    log_s, targets, categories = _instance(seed)
    fast = calibrated_lists(log_s, targets, categories, lam, 6)
    for row in range(log_s.shape[0]):
        chosen = []
        for _ in range(6):
            cands = [i for i in range(log_s.shape[1]) if i not in chosen]
            values = [_objective(np.array(chosen + [i]), log_s[row], targets[row], categories, lam) for i in cands]
            chosen.append(cands[int(np.argmax(values))])
        assert fast[row].tolist() == chosen


def test_strong_calibration_reproduces_target_proportions():
    """Equal scores, target 70:30 over two of three categories: a 10-item list takes 7 and 3."""
    categories = np.repeat(np.arange(3), 20)
    lists = calibrated_lists(np.zeros((1, 60)), np.array([[0.7, 0.3, 0.0]]), categories, 0.99, 10)
    assert np.bincount(categories[lists[0]], minlength=3).tolist() == [7, 3, 0]


def test_lists_are_distinct_and_validated():
    log_s, targets, categories = _instance(0)
    lists = calibrated_lists(log_s, targets, categories, 0.9, 20)
    assert all(len(set(row)) == 20 for row in lists.tolist())
    with pytest.raises(ValueError):
        calibrated_lists(log_s, targets, categories, 1.5, 5)
