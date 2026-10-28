"""Provisional estimation metrics, hand-checked."""
import numpy as np

from adaptdiv.estimation_metrics import crossover_lag, false_switch_rate, noise_floor, tracking_error, tv

A = np.array([1.0, 0.0])
B = np.array([0.0, 1.0])


def test_tv_basic():
    assert np.isclose(tv(A, B), 1.0)
    assert np.isclose(tv(np.array([0.5, 0.5]), A), 0.5)


def test_tracking_error_alignment():
    z = np.array([A, A, B])                                         # z_1, z_2, z_3
    zhat = np.array([[0.5, 0.5], A, [0.5, 0.5], B])                 # zhat_0 .. zhat_3
    assert np.allclose(tracking_error(zhat, z), [0.0, 0.5, 0.0])   # zhat_0 is never scored


def test_noise_floor():
    z = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.isclose(noise_floor(z, 0.2), 0.2 * 0.75)


def _zhat(p_new):
    """Estimates as a mix (1-p) A + p B, with zhat_0 = A."""
    p = np.concatenate([[0.0], p_new])
    return (1 - p)[:, None] * A + p[:, None] * B


def test_crossover_lag_hand_example():
    # start = 3: zhat_3.. = 0.2, 0.6, 0.4, 0.7, 0.8, 0.9, 0.9, 0.9 -> closer to B from t=6 onward
    zhat = _zhat(np.array([0, 0, 0.2, 0.6, 0.4, 0.7, 0.8, 0.9, 0.9, 0.9]))
    lag, cens = crossover_lag(zhat, A, B, start=3, run=3)
    assert lag == 3 and not cens  # t = 6 = start + 3; the brief excursion at t=4 does not count


def test_crossover_lag_censored():
    zhat = _zhat(np.array([0, 0, 0.2, 0.6, 0.4, 0.3]))
    lag, cens = crossover_lag(zhat, A, B, start=3, run=2)
    assert cens and lag == 6 - 3 + 1


def test_crossover_lag_batched():
    z1 = _zhat(np.array([0, 0, 0.9, 0.9, 0.9, 0.9]))
    z2 = _zhat(np.array([0, 0, 0.1, 0.1, 0.9, 0.9]))
    lag, cens = crossover_lag(np.stack([z1, z2]), np.stack([A, A]), np.stack([B, B]), start=3, run=2)
    assert list(lag) == [0, 2] and not cens.any()


def test_false_switch_rate():
    z = np.tile(A, (6, 1))                                  # true top never changes
    tops = [0, 0, 1, 1, 0, 0]                               # estimate flips at t=3 and t=5
    zhat = np.vstack([[0.5, 0.5], np.eye(2)[tops]])
    assert np.isclose(false_switch_rate(zhat, z, burn_in=0), 2 / 5)
    assert np.isclose(false_switch_rate(zhat, z, burn_in=3), 1 / 3)  # only t = 4, 5, 6 counted


def test_true_switch_is_not_false():
    z = np.array([A, A, B, B])
    zhat = np.vstack([[0.5, 0.5], A, A, B, B])               # estimate switches exactly when truth does
    assert false_switch_rate(zhat, z, burn_in=0) == 0.0
