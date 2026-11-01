"""Milestone 6 DPP check (exploratory; TUNING seeds only; not a hypothesis test).

1. Greedy quality: greedy vs brute force on tiny instances built from real embeddings.
2. Speed: milliseconds per list at full size.
3. Theta sweep: Top-K and DPP for static, adaptive and oracle profiles, measuring next-item
   coverage, categories per list, intra-list diversity and already-clicked share.

Uses a subsample (first 30 users of each cell, every 5th step) to keep the run to a few
minutes. Metrics are provisional until M7 (D-17). No theta is selected here.

    python experiments/dpp_check.py
Writes results/m6_dpp/<run_id>/{config.json, check.json, environment.json, tradeoff.png}.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from adaptdiv.config import TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.dpp import brute_force_map, greedy_map, objective  # noqa: E402
from adaptdiv.estimators import PROPOSED_ESTIMATOR, STATIC_ESTIMATOR, categories_of  # noqa: E402
from adaptdiv.recommenders import log_scores, popularity_list, recommendation_lists  # noqa: E402
from adaptdiv.relevance import global_popularity, within_category_popularity  # noqa: E402
from adaptdiv.similarity import similarity_matrix  # noqa: E402
from adaptdiv.simulator import generate  # noqa: E402
from estimator_study import BURN_IN, POST, switch_events  # noqa: E402

K = 10
USERS = 30
STRIDE = 5
THETAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)   # 1.0 = Top-K
PROFILES = ("static", "adaptive", "oracle")


def greedy_quality(n_instances: int = 200, n: int = 14, k: int = 5) -> dict:
    """Greedy vs exact on tiny instances from a real catalogue (rng local to this check)."""
    cfg = SimulationConfig()
    data = generate(TUNING_SEEDS[0], replace(cfg, population=replace(cfg.population, users_per_cell=1, train_users=10)), scenarios=("stable",))
    phi = data.catalogue.embeddings
    rng = np.random.default_rng(2026)
    out = {}
    for theta in (0.3, 0.6, 0.8):
        gaps, exact_hits = [], 0
        for _ in range(n_instances):
            idx = rng.choice(phi.shape[0], n, replace=False)
            S = similarity_matrix(phi[idx])
            log_s = np.log(rng.dirichlet(np.ones(n)))
            g = objective(greedy_map(log_s, S, theta, k), log_s, S, theta)
            b = objective(brute_force_map(log_s, S, theta, k), log_s, S, theta)
            gaps.append(b - g)
            exact_hits += gaps[-1] < 1e-9
        out[str(theta)] = {"fraction_exact": exact_hits / n_instances, "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps))}
    return out


def speed(S: np.ndarray, r: np.ndarray, categories: np.ndarray) -> float:
    z = np.random.default_rng(1).dirichlet(np.ones(10), 2000)
    ls = log_scores(z, r, categories)
    t0 = time.perf_counter()
    for row in ls:
        greedy_map(row, S, 0.6, K)
    return 1000 * (time.perf_counter() - t0) / len(ls)


def list_metrics(lists, truth, items, phi, categories, steps) -> dict:
    C = truth.z.shape[1]
    n_c = truth.item_conditional.shape[1]
    mass = ((1 - truth.noise_rate) * truth.z + truth.noise_rate / C)[steps]
    cats, local = lists // n_c, lists % n_c
    cov = (np.take_along_axis(mass, cats, axis=1) * truth.item_conditional[cats, local]).sum(axis=1)
    n_cats = np.array([len(set(row)) for row in cats])
    E = phi[lists]                                            # (n, K, d)
    sims = np.einsum("nid,njd->nij", E, E)
    ild = 1.0 - (sims.sum(axis=(1, 2)) - K) / (K * (K - 1))
    seen = np.array([np.isin(row, items[:t]).mean() for row, t in zip(lists, steps)])
    return {"coverage": cov, "categories": n_cats, "ild": ild, "seen": seen}


def main() -> None:
    cfg = SimulationConfig()
    cfg = replace(cfg, population=replace(cfg.population, users_per_cell=USERS))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / "m6_dpp" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out / "config.json")
    T = cfg.scenario.horizon
    steps = np.arange(BURN_IN, T, STRIDE)   # row t = list for step t + 1, built from zhat_t

    quality = greedy_quality()
    print("greedy vs exact:", quality)

    rows: dict[str, dict] = {}
    speeds = []
    t_start = time.perf_counter()
    for seed in TUNING_SEEDS:
        data = generate(seed, cfg)
        pub = data.catalogue.public()
        S = similarity_matrix(pub.embeddings)
        r = within_category_popularity(data.train_item_counts, pub.categories, pub.n_categories)
        pop = popularity_list(global_popularity(data.train_item_counts), K)
        speeds.append(speed(S, r, pub.categories))

        for (kappa, scen), users in data.users.items():
            starts = [s for s, _, _ in switch_events(scen, cfg)]
            post = np.zeros(len(steps), dtype=bool)
            for s in starts:
                post |= (steps >= s - 1) & (steps < s - 1 + POST)
            phase = post if starts else np.ones(len(steps), dtype=bool)

            cats = np.stack([categories_of(u.observed.items, pub.categories) for u in users])
            prof = {
                "static": STATIC_ESTIMATOR(cats, pub.n_categories)[:, :T],
                "adaptive": PROPOSED_ESTIMATOR(cats, pub.n_categories)[:, :T],
                "oracle": np.stack([u.truth.z for u in users]),   # diagnostic ceiling only
            }
            def record(name, per_user):
                entry = rows.setdefault(f"kappa={kappa}|{scen}|{name}", {m: [] for m in per_user[0]})
                for m in entry:
                    entry[m].append(float(np.mean([p[m][phase].mean() for p in per_user])))

            record("popularity", [list_metrics(np.tile(pop, (len(steps), 1)), u.truth, u.observed.items,
                                               pub.embeddings, pub.categories, steps) for u in users])
            for p in PROFILES:
                for th in THETAS:
                    per_user = []
                    for j, u in enumerate(users):
                        lists = recommendation_lists(prof[p][j], r, pub.categories, S, th, K, steps=steps)
                        per_user.append(list_metrics(lists, u.truth, u.observed.items, pub.embeddings, pub.categories, steps))
                    record(f"{p}|theta={th}", per_user)
        print(f"seed {seed} done ({time.perf_counter() - t_start:.0f}s)")

    (out / "check.json").write_text(json.dumps(
        {"seeds": list(TUNING_SEEDS), "k": K, "users_per_cell": USERS, "stride": STRIDE, "thetas": THETAS,
         "phase": "stable/noisy: steps after burn-in; drift: 50 steps after each switch",
         "greedy_quality": quality, "ms_per_list": speeds, "runtime_seconds": time.perf_counter() - t_start,
         "cells": rows}, indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")
    plot(out, rows, cfg)
    print(f"ms per list (N=500, K=10): {np.mean(speeds):.3f}")
    print(f"wrote {out}")


def plot(out: Path, rows: dict, cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"static": "#2a78d6", "adaptive": "#eb6834", "oracle": "#52514e"}
    labels = {"static": "Static profile", "adaptive": "Adaptive profile", "oracle": "Oracle (true interests)"}
    scens = ("stable", "abrupt")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="#fcfcfb")
    for r, kappa in enumerate(cfg.population.kappas):
        for c, scen in enumerate(scens):
            ax = axes[r, c]
            ax.set_facecolor("#fcfcfb")
            for p in PROFILES:
                xs = [np.mean(rows[f"kappa={kappa}|{scen}|{p}|theta={t}"]["categories"]) for t in THETAS]
                ys = [np.mean(rows[f"kappa={kappa}|{scen}|{p}|theta={t}"]["coverage"]) for t in THETAS]
                ax.plot(xs, ys, color=colours[p], lw=2, marker="o", ms=4, ls="--" if p == "oracle" else "-", label=labels[p])
                ax.plot(xs[0], ys[0], marker="o", ms=10, mfc="none", mec=colours[p], mew=1.5, ls="none")
            ax.grid(color="#e6e5e0", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{'Focused' if kappa == 0.1 else 'Spread'} users, {scen}" + (" (50 steps after shift)" if scen == "abrupt" else ""),
                         fontsize=10, loc="left", color="#0b0b0b")
            ax.set_xlabel("distinct categories per list (K = 10)", fontsize=8, color="#52514e")
            ax.set_ylabel("next-click probability covered", fontsize=8, color="#52514e")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Relevance-diversity trade-off as theta falls from 1.0 (Top-K, ringed point) to 0.3; tuning seeds; y-axes differ",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
