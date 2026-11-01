"""Statistics (hand-checked) and the frozen analysis on tables with planted effects."""
import numpy as np

from adaptdiv.confirmatory import (
    ALL_SCENARIOS,
    KAPPAS,
    analyse,
    confirmatory_tests,
    h4b_equivalence,
    h6a_dominance,
    pooled,
    stack_records,
)
from adaptdiv.stats import bootstrap_ci, holm, wilcoxon_p

N = 20
DRIFT = ("abrupt", "gradual", "recurrent")
METHODS = ("popularity", "static_topk", "adaptive_topk", "static_dpp", "adaptive_dpp", "oracle_dpp")


def test_holm_hand_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_wilcoxon_exact_small_sample():
    assert np.isclose(wilcoxon_p(np.array([1.0, 2, 3, 4, 5, 6])), 2 / 2 ** 6)
    assert wilcoxon_p(np.zeros(5)) == 1.0


def test_bootstrap_ci():
    assert bootstrap_ci(np.full(10, 2.0)) == (2.0, 2.0)
    d = np.random.default_rng(0).normal(1.0, 0.2, 20)
    low, high = bootstrap_ci(d)
    assert low < d.mean() < high
    assert bootstrap_ci(d) == (low, high)  # deterministic


def _table(effect: float, seed=0):
    """Seed-level table. effect = 0 gives no differences beyond noise; effect = 1 plants H1-H6 effects."""
    rng = np.random.default_rng(seed)
    noise = lambda: rng.normal(0, 0.002, N)  # noqa: E731
    base = rng.normal(0, 0.01, N)  # shared seed effect, as with paired simulations
    t = {}
    for k in KAPPAS:
        for s in ALL_SCENARIOS:
            for name, cum, ewma in (("tracking", 0.2, 0.2), ("false_switch", 0.05, 0.05)):
                up = effect * (0.05 if s in ("stable", "noisy") else -0.1)
                t[(k, s, "est:cumulative", "overall", name)] = cum + base + noise()
                t[(k, s, "est:ewma", "overall", name)] = ewma + up + base + noise()
            if s in DRIFT:
                t[(k, s, "est:cumulative", "post", "lag")] = 30 + 100 * base + noise()
                t[(k, s, "est:ewma", "post", "lag")] = 30 - effect * 20 + 100 * base + noise()
                t[(k, s, "est:cumulative", "post", "tracking")] = 0.5 + base + noise()
                t[(k, s, "est:ewma", "post", "tracking")] = 0.5 - effect * 0.2 + base + noise()
            for m in METHODS:
                adaptive = "adaptive" in m
                dpp = "dpp" in m
                cost = 0.02 if k == 0.1 else 0.01
                for phase in ("overall", "post"):
                    gain = effect * 0.1 if (adaptive and phase == "post" and s in DRIFT) else 0.0
                    if dpp and effect:
                        gain = gain * 0.5 - cost
                    t[(k, s, m, phase, "ndcg")] = 0.3 + gain + base + noise()
                    t[(k, s, m, phase, "coverage")] = 0.3 + (0.1 * effect if dpp else 0.0) + base + noise()
    return t


def test_planted_effects_are_all_found_in_the_right_direction():
    results = {r["test"].split()[0]: r for r in confirmatory_tests(_table(1.0))}
    assert len(results) == 8
    for name in ("H1a", "H1b", "H2", "H3", "H4a", "H5a", "H5b"):
        assert results[name]["supported"], name
    assert results["H6b"]["significant"] and results["H6b"]["direction"] == "negative"  # substitution planted
    assert results["H1a"]["practically_meaningful"] and results["H1b"]["practically_meaningful"]


def test_family_wise_false_positive_rate_is_controlled_on_null_tables():
    """With no planted effects, Holm should keep P(any significant) near 5%, and a chance
    significant result must not count as practically meaningful."""
    any_significant = 0
    for seed in range(200):
        results = confirmatory_tests(_table(0.0, seed))
        any_significant += any(r["significant"] for r in results)
        assert not any(r["practically_meaningful"] for r in results)
    assert any_significant / 200 < 0.10


def test_wrong_direction_is_not_supported():
    t = _table(1.0)
    for k in KAPPAS:
        for s in ALL_SCENARIOS:
            key_e, key_c = (k, s, "est:ewma", "overall", "tracking"), (k, s, "est:cumulative", "overall", "tracking")
            t[key_e], t[key_c] = t[key_c], t[key_e]   # H2 reversed
    r = {x["test"].split()[0]: x for x in confirmatory_tests(t)}
    assert r["H2"]["significant"] and not r["H2"]["supported"]


def test_small_effect_is_significant_but_not_practically_meaningful():
    t = _table(0.0)
    for k in KAPPAS:
        for s in DRIFT:
            t[(k, s, "est:ewma", "post", "tracking")] = t[(k, s, "est:cumulative", "post", "tracking")] - 0.01
    r = {x["test"].split()[0]: x for x in confirmatory_tests(t)}
    assert r["H1b"]["supported"] and r["H1b"]["practically_meaningful"] is False


def test_h4b_equivalence():
    assert h4b_equivalence(_table(0.0))["equivalent"]
    t = _table(0.0)
    for k in KAPPAS:
        t[(k, "stable", "adaptive_topk", "overall", "ndcg")] = t[(k, "stable", "static_topk", "overall", "ndcg")] * 1.2
    assert not h4b_equivalence(t)["equivalent"]


def test_h6a_detects_a_dominator_and_ignores_oracles():
    t = _table(0.0)
    for k in KAPPAS:
        for s in DRIFT:
            t[(k, s, "adaptive_dpp", "post", "ndcg")] = t[(k, s, "oracle_dpp", "post", "ndcg")] - 0.05
            t[(k, s, "adaptive_dpp", "post", "coverage")] = t[(k, s, "oracle_dpp", "post", "coverage")] - 0.05
    assert h6a_dominance(t)["dominated_by"] == ["popularity", "static_topk", "adaptive_topk", "static_dpp"]
    t2 = _table(1.0)
    for k in KAPPAS:
        for s in DRIFT:
            t2[(k, s, "oracle_dpp", "post", "ndcg")] += 1.0
            t2[(k, s, "oracle_dpp", "post", "coverage")] += 1.0
    assert t2 and "oracle_dpp" not in h6a_dominance(t2)["dominated_by"]


def test_pooled_is_equal_weight():
    t = _table(0.0)
    manual = np.mean([t[(k, s, "static_topk", "post", "ndcg")] for k in KAPPAS for s in DRIFT], axis=0)
    assert np.allclose(pooled(t, "static_topk", "post", "ndcg"), manual)


def test_analyse_and_stack_records():
    assert set(analyse(_table(1.0))) == {"tests", "h4b", "h6a"}
    stacked = stack_records([{"a": 1.0}, {"a": 2.0}])
    assert stacked["a"].tolist() == [1.0, 2.0]
