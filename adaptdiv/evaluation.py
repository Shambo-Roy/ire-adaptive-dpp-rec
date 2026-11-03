"""Evaluate methods on one seed's simulated data (DECISIONS.md D-17, D-30).

Information flow, per user:
  observed items -> categories -> profiles -> lists      (no ground truth, except labelled oracle)
  lists + held-out items + GroundTruth -> metrics        (evaluation only)

The result is a flat dict of seed-level values:
  records[(kappa, scenario, method, phase, metric)] = value
Phases: "overall" (steps after burn-in) and "post" (windows after each switch; drift scenarios).
Estimators are recorded as methods "est:cumulative" and "est:ewma".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics as M
from .calibration import miscalibration
from .config import SimulationConfig
from .estimation_metrics import crossover_lag, false_switch_rate, tracking_error
from .estimators import PROPOSED_ESTIMATOR, STATIC_ESTIMATOR, UNIFORM_START_ABLATION, EstimatorSpec, categories_of
from .recommenders import Method, log_scores, popularity_list, recommendation_lists
from .relevance import global_popularity, within_category_popularity
from .similarity import similarity_matrix
from .simulator import SimulatedData, SimulatedUser, generate

Records = dict[tuple[float, str, str, str, str], float]

RELEVANCE_METRICS = ("ndcg", "recall", "eu", "ndcg_new", "eu_new", "coverage", "ild", "seen", "miscal")


@dataclass(frozen=True)
class EvalConfig:
    k: int = 10             # list size
    h: int = 5              # held-out clicks per list
    burn_in: int = 20       # lists start after this many observations
    post_window: int = 50   # steps after each switch
    crossover_run: int = 5  # consecutive steps for the crossover lag
    stride: int = 1         # score every stride-th step


def switch_events(scenario: str, cfg: SimulationConfig) -> list[tuple[int, str, str]]:
    """(start step, old, new) for each shift. Gradual uses the ramp midpoint, where truth is equidistant."""
    sc = cfg.scenario
    if scenario == "abrupt":
        return [(sc.abrupt_switch, "z_a", "z_b")]
    if scenario == "gradual":
        return [(sc.gradual_start + sc.gradual_width // 2, "z_a", "z_b")]
    if scenario == "recurrent":
        t1, t2 = sc.recurrent_switches
        return [(t1, "z_a", "z_b"), (t2, "z_b", "z_a")]
    return []


def scored_steps(T: int, ev: EvalConfig) -> np.ndarray:
    """History lengths t with a full target window: burn_in .. T - h."""
    return np.arange(ev.burn_in, T - ev.h + 1, ev.stride)


def post_mask(steps: np.ndarray, starts: list[int], window: int) -> np.ndarray:
    """Lists with s - 1 <= t < s - 1 + window: the first target of t = s - 1 is step s."""
    mask = np.zeros(steps.shape[0], dtype=bool)
    for s in starts:
        mask |= (steps >= s - 1) & (steps < s - 1 + window)
    return mask


def build_profiles(cats: np.ndarray, n_categories: int, oracle_z: np.ndarray | None) -> dict[str, np.ndarray]:
    """Profiles (U, T, C): row t = estimate from the first t observations.

    ``oracle_z`` is the true interest (row t = z_{t+1}); passing it is the only way ground
    truth reaches a list, and only the labelled oracle diagnostics read that entry.
    """
    T = cats.shape[1]
    prof = {
        "static": STATIC_ESTIMATOR(cats, n_categories)[:, :T],
        "adaptive": PROPOSED_ESTIMATOR(cats, n_categories)[:, :T],
        "uniform_start": UNIFORM_START_ABLATION(cats, n_categories)[:, :T],
        "window": EstimatorSpec("window", 40)(cats, n_categories)[:, :T],
    }
    if oracle_z is not None:
        prof["oracle"] = oracle_z
    return prof


def list_metrics(lists: np.ndarray, steps: np.ndarray, user: SimulatedUser, public, ev: EvalConfig) -> dict[str, np.ndarray]:
    """All D-17 list metrics for one user; each array has one value per list."""
    items, truth = user.observed.items, user.truth
    n_c = truth.item_conditional.shape[1]
    C = public.n_categories
    targets = M.target_matrix(items, steps, ev.h)

    seen = np.zeros((items.shape[0] + 1, public.categories.shape[0]), dtype=bool)
    seen[np.arange(1, items.shape[0] + 1), items] = True
    seen = np.logical_or.accumulate(seen, axis=0)[steps]     # row n: clicked in i_1..i_t
    list_seen = np.take_along_axis(seen, lists, axis=1)
    target_new = ~np.take_along_axis(seen, targets, axis=1)

    hit = M.hits(lists, targets)
    n_rel = M.n_targets(targets)
    hit_new = M.hits(lists, targets, target_new)
    n_new = M.n_targets(targets, target_new)

    mass = ((1 - truth.noise_rate) * truth.z + truth.noise_rate / C)[steps]
    cats, local = lists // n_c, lists % n_c
    probs = np.take_along_axis(mass, cats, axis=1) * truth.item_conditional[cats, local]

    return {
        "ndcg": M.ndcg_at_k(hit, n_rel),
        "recall": M.recall_at_k(hit, n_rel),
        "eu": M.expected_utility(probs),
        "ndcg_new": M.ndcg_at_k(hit_new, n_new),
        "eu_new": M.expected_utility(probs, ~list_seen),
        "coverage": M.category_coverage(public.categories[lists], C),
        "ild": M.intra_list_diversity(public.embeddings[lists]),
        "seen": list_seen.mean(axis=1),
        "miscal": miscalibration(public.categories[lists], truth.z[steps], C),   # D-19: KL(z_{t+1} || q~)
    }


def method_lists(method: Method, profile_row_user, r_within, pop_list, public, S, steps, k) -> np.ndarray:
    """Lists for one user, sorted by score (a DPP selects a set; D-17)."""
    if method.profile == "none":
        return np.tile(pop_list, (steps.shape[0], 1))
    lists = recommendation_lists(profile_row_user, r_within, public.categories, S, method.theta, k, steps=steps,
                                 calib_lambda=method.calib_lambda)
    log_s = log_scores(profile_row_user[steps], r_within, public.categories)
    order = np.argsort(-np.take_along_axis(log_s, lists, axis=1), axis=1, kind="stable")
    return np.take_along_axis(lists, order, axis=1)


def estimator_records(records: Records, kappa, scen, cats, cell, cfg: SimulationConfig, ev: EvalConfig) -> None:
    C = cell["z"].shape[-1]
    for name, spec in (("est:cumulative", STATIC_ESTIMATOR), ("est:ewma", PROPOSED_ESTIMATOR)):
        zhat = spec(cats, C)
        err = tracking_error(zhat, cell["z"])
        records[(kappa, scen, name, "overall", "tracking")] = float(err[:, ev.burn_in:].mean())
        records[(kappa, scen, name, "overall", "false_switch")] = float(false_switch_rate(zhat, cell["z"], ev.burn_in).mean())
        events = switch_events(scen, cfg)
        if events:
            lags, post = [], []
            for start, old, new in events:
                lag, _ = crossover_lag(zhat, cell[old], cell[new], start, ev.crossover_run)
                lags.append(lag)
                post.append(err[:, start - 1: start - 1 + ev.post_window].mean(axis=1))
            records[(kappa, scen, name, "post", "lag")] = float(np.mean(lags))
            records[(kappa, scen, name, "post", "tracking")] = float(np.mean(post))


def evaluate(
    data: SimulatedData,
    methods: tuple[Method, ...],
    ev: EvalConfig = EvalConfig(),
    extra_profiles: dict[str, EstimatorSpec] | None = None,
) -> Records:
    """Seed-level records for all cells of one seed. ``extra_profiles`` adds estimators by profile name."""
    cfg = data.config
    public = data.catalogue.public()
    S = similarity_matrix(public.embeddings)
    r_within = within_category_popularity(data.train_item_counts, public.categories, public.n_categories)
    pop_list = popularity_list(global_popularity(data.train_item_counts), ev.k)
    steps = scored_steps(cfg.scenario.horizon, ev)

    records: Records = {}
    for (kappa, scen), users in data.users.items():
        cats = np.stack([categories_of(u.observed.items, public.categories) for u in users])
        z = np.stack([u.truth.z for u in users])
        cell = {"z": z, "z_a": np.stack([u.truth.params.z_a for u in users]), "z_b": np.stack([u.truth.params.z_b for u in users])}
        estimator_records(records, kappa, scen, cats, cell, cfg, ev)

        needs_oracle = any(m.profile == "oracle" for m in methods)
        profiles = build_profiles(cats, public.n_categories, z if needs_oracle else None)
        for name, spec in (extra_profiles or {}).items():
            profiles[name] = spec(cats, public.n_categories)[:, :cats.shape[1]]
        starts = [s for s, _, _ in switch_events(scen, cfg)]
        phases = {"overall": np.ones(steps.shape[0], dtype=bool)}
        if starts:
            phases["post"] = post_mask(steps, starts, ev.post_window)

        for method in methods:
            per_user = {p: {m: [] for m in RELEVANCE_METRICS} for p in phases}
            for j, user in enumerate(users):
                prof = profiles[method.profile][j] if method.profile != "none" else None
                lists = method_lists(method, prof, r_within, pop_list, public, S, steps, ev.k)
                values = list_metrics(lists, steps, user, public, ev)
                for p, mask in phases.items():
                    for m in RELEVANCE_METRICS:
                        v = values[m][mask]
                        per_user[p][m].append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
            for p in phases:
                for m in RELEVANCE_METRICS:
                    records[(kappa, scen, method.name, p, m)] = float(np.nanmean(per_user[p][m]))
    return records


def evaluate_seed(
    seed: int,
    cfg: SimulationConfig,
    methods: tuple[Method, ...],
    ev: EvalConfig = EvalConfig(),
    extra_profiles: dict[str, EstimatorSpec] | None = None,
) -> Records:
    return evaluate(generate(seed, cfg), methods, ev, extra_profiles)
