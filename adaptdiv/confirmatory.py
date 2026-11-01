"""Frozen confirmatory analysis (DECISIONS.md D-21, D-30; hypothesis.md).

Input: ``table[(kappa, scenario, method, phase, metric)]`` = array of seed-level values, in one
fixed seed order. Eight tests run under Holm correction; H4b and H6a are separate checks.

"Supported" means: Holm-adjusted p < 0.05 and the mean difference has the predicted sign.
"Practically meaningful" means the whole 95% CI lies beyond the SESOI, where one is defined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import bootstrap_ci, holm, wilcoxon_p

ALPHA = 0.05
KAPPAS = (0.1, 1.0)
DRIFT = ("abrupt", "gradual", "recurrent")
ALL_SCENARIOS = ("stable", "abrupt", "gradual", "recurrent", "noisy")
SESOI_RELATIVE = 0.03   # NDCG / expected utility
SESOI_LAG = 2.0         # interactions
SESOI_TV = 0.02         # absolute tracking error


@dataclass(frozen=True)
class Names:
    """Method names in the table."""

    static_topk: str = "static_topk"
    adaptive_topk: str = "adaptive_topk"
    static_dpp: str = "static_dpp"
    adaptive_dpp: str = "adaptive_dpp"
    popularity: str = "popularity"
    ewma: str = "est:ewma"
    cumulative: str = "est:cumulative"


def pooled(table, method: str, phase: str, metric: str, scenarios=DRIFT, kappas=KAPPAS) -> np.ndarray:
    """Equal-weight mean over the given cells; one value per seed."""
    return np.mean([table[(k, s, method, phase, metric)] for k in kappas for s in scenarios], axis=0)


def _test(name, diff, predicted, sesoi=None, baseline=None) -> dict:
    """``predicted``: +1, or 0 for two-sided. ``baseline`` turns the SESOI into a relative one."""
    diff = np.asarray(diff, dtype=float)
    rel = diff / baseline if baseline is not None else diff
    low, high = bootstrap_ci(rel)
    return {
        "test": name,
        "mean_diff": float(diff.mean()),
        "effect_for_sesoi": float(rel.mean()),
        "ci_low": low,
        "ci_high": high,
        "p": wilcoxon_p(diff),
        "predicted_sign": predicted,
        "sesoi": sesoi,
    }


def _relative_cost(table, names: Names, kappa: float) -> np.ndarray:
    dpp = pooled(table, names.adaptive_dpp, "overall", "ndcg", ALL_SCENARIOS, (kappa,))
    top = pooled(table, names.adaptive_topk, "overall", "ndcg", ALL_SCENARIOS, (kappa,))
    return 1.0 - dpp / top


def confirmatory_tests(table, names: Names = Names()) -> list[dict]:
    """The eight D-30 tests with Holm-adjusted p-values and verdicts."""
    n = names
    ndcg = lambda m: pooled(table, m, "post", "ndcg")  # noqa: E731
    stat, adat, sdpp, adpp = ndcg(n.static_topk), ndcg(n.adaptive_topk), ndcg(n.static_dpp), ndcg(n.adaptive_dpp)
    tests = [
        _test("H1a lag: cumulative - EWMA",
              pooled(table, n.cumulative, "post", "lag") - pooled(table, n.ewma, "post", "lag"), +1, SESOI_LAG),
        _test("H1b post-shift tracking: cumulative - EWMA",
              pooled(table, n.cumulative, "post", "tracking") - pooled(table, n.ewma, "post", "tracking"), +1, SESOI_TV),
        _test("H2 stable tracking: EWMA - cumulative",
              pooled(table, n.ewma, "overall", "tracking", ("stable",)) - pooled(table, n.cumulative, "overall", "tracking", ("stable",)),
              +1, SESOI_TV),
        _test("H3 noisy false switches: EWMA - cumulative",
              pooled(table, n.ewma, "overall", "false_switch", ("noisy",)) - pooled(table, n.cumulative, "overall", "false_switch", ("noisy",)),
              +1),
        _test("H4a post-shift NDCG: Adaptive Top-K - Static Top-K", adat - stat, +1, SESOI_RELATIVE, stat),
        _test("H5a coverage: Adaptive DPP - Adaptive Top-K",
              pooled(table, n.adaptive_dpp, "overall", "coverage", ALL_SCENARIOS) - pooled(table, n.adaptive_topk, "overall", "coverage", ALL_SCENARIOS),
              +1),
        _test("H5b relative NDCG cost of DPP: focused - spread",
              _relative_cost(table, n, KAPPAS[0]) - _relative_cost(table, n, KAPPAS[1]), +1),
        _test("H6b adaptation x selector interaction on post-shift NDCG", (adpp - sdpp) - (adat - stat), 0),
    ]
    for t, adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = adj
        sign_ok = t["predicted_sign"] == 0 or np.sign(t["mean_diff"]) == t["predicted_sign"]
        t["significant"] = bool(adj < ALPHA)
        t["supported"] = bool(adj < ALPHA and sign_ok)
        t["direction"] = "positive" if t["mean_diff"] > 0 else "negative"
        if t["sesoi"] is None:
            t["practically_meaningful"] = None
        else:
            t["practically_meaningful"] = bool(t["ci_low"] > t["sesoi"] or t["ci_high"] < -t["sesoi"])
    return tests


def h4b_equivalence(table, names: Names = Names()) -> dict:
    """Equivalence on stable interests: the 95% CI of the relative difference lies within +/- SESOI."""
    ada = pooled(table, names.adaptive_topk, "overall", "ndcg", ("stable",))
    sta = pooled(table, names.static_topk, "overall", "ndcg", ("stable",))
    low, high = bootstrap_ci((ada - sta) / sta)
    return {"test": "H4b stable equivalence", "ci_low": low, "ci_high": high, "equivalent": bool(low > -SESOI_RELATIVE and high < SESOI_RELATIVE)}


def h6a_dominance(table, names: Names = Names()) -> dict:
    """Adaptive DPP is dominated if a compared method beats it on both axes, with both CIs above 0."""
    def axes(m):
        return pooled(table, m, "post", "ndcg"), pooled(table, m, "post", "coverage")

    ndcg_a, cov_a = axes(names.adaptive_dpp)
    dominators, detail = [], {}
    for m in (names.popularity, names.static_topk, names.adaptive_topk, names.static_dpp):  # oracles excluded (D-30)
        ndcg_b, cov_b = axes(m)
        ci_n, ci_c = bootstrap_ci(ndcg_b - ndcg_a), bootstrap_ci(cov_b - cov_a)
        detail[m] = {"ndcg_ci": ci_n, "coverage_ci": ci_c}
        if ci_n[0] > 0 and ci_c[0] > 0:
            dominators.append(m)
    return {"test": "H6a not dominated", "dominated_by": dominators, "not_dominated": not dominators, "detail": detail}


def analyse(table, names: Names = Names()) -> dict:
    return {"tests": confirmatory_tests(table, names), "h4b": h4b_equivalence(table, names), "h6a": h6a_dominance(table, names)}


def stack_records(per_seed: list[dict]) -> dict:
    """Turn a list of per-seed records into {key: array over seeds}; keys must match across seeds."""
    keys = per_seed[0].keys()
    return {k: np.array([rec[k] for rec in per_seed]) for k in keys}
