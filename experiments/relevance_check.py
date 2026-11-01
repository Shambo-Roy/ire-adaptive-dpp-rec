"""Milestone 5 scoring check (exploratory; TUNING seeds only; not a hypothesis test).

Ranks all items with Top-K on plug-in scores and measures how much of the true next-item
probability the list covers: sum over list R_t of P_{t+1}(i). This is a sanity check of
the scores before the DPP exists. Formal metrics are fixed at M7 (D-17).

Timing: a list for step t + 1 uses only zhat_t (the first t observations).
The oracle row uses the true z_{t+1}: a diagnostic ceiling, never a competing method (D-16).

    python experiments/relevance_check.py
Writes results/m5_relevance/<run_id>/{config.json, check.json, environment.json}.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from adaptdiv.config import SCENARIOS, TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.estimators import PROPOSED_ESTIMATOR, STATIC_ESTIMATOR, categories_of  # noqa: E402
from adaptdiv.relevance import global_popularity, plugin_scores, top_k, within_category_popularity  # noqa: E402
from adaptdiv.similarity import min_eigenvalue, similarity_matrix  # noqa: E402
from adaptdiv.simulator import generate  # noqa: E402
from estimator_study import BURN_IN, POST, switch_events  # noqa: E402

K = 10
METHODS = ("popularity", "static", "adaptive", "oracle_z")


def coverage(lists: np.ndarray, truth) -> np.ndarray:
    """Sum of P_{t+1}(i) over each list; ``lists`` is (T, K), the result (T,)."""
    C = truth.z.shape[1]
    cat_mass = (1 - truth.noise_rate) * truth.z + truth.noise_rate / C        # (T, C)
    n_c = truth.item_conditional.shape[1]
    cats, local = lists // n_c, lists % n_c
    return (np.take_along_axis(cat_mass, cats, axis=1) * truth.item_conditional[cats, local]).sum(axis=1)


def seen_fraction(lists: np.ndarray, items: np.ndarray, n_items: int) -> np.ndarray:
    """For each step t + 1, the fraction of the list already clicked in steps 1..t."""
    T = items.shape[0]
    seen = np.zeros((T, n_items), dtype=bool)
    rows = np.arange(1, T)
    seen[rows, items[:-1]] = True
    seen = np.logical_or.accumulate(seen, axis=0)
    return np.take_along_axis(seen, lists, axis=1).mean(axis=1)


def run_cell(users, public, r_within, r_global, T: int) -> dict:
    C = public.n_categories
    cats = np.stack([categories_of(u.observed.items, public.categories) for u in users])
    profiles = {
        "static": STATIC_ESTIMATOR(cats, C)[:, :T],      # row t = zhat_t, used for step t + 1
        "adaptive": PROPOSED_ESTIMATOR(cats, C)[:, :T],
        "oracle_z": np.stack([u.truth.z for u in users]),  # diagnostic: true z_{t+1}
    }
    out = {m: [] for m in METHODS}
    pop_list = np.broadcast_to(top_k(r_global, K), (T, K))
    repeats = []
    for j, u in enumerate(users):
        out["popularity"].append(coverage(pop_list, u.truth))
        for m, prof in profiles.items():
            lists = top_k(plugin_scores(prof[j], r_within, public.categories), K)
            out[m].append(coverage(lists, u.truth))
            if m == "adaptive":
                repeats.append(seen_fraction(lists, u.observed.items, public.categories.shape[0]))
    return {"coverage": {m: np.stack(v) for m, v in out.items()}, "seen": np.stack(repeats)}


def main() -> None:
    cfg = SimulationConfig()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / "m5_relevance" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "config.json")
    T = cfg.scenario.horizon

    rows: dict[str, dict] = {}
    kernel = []
    for seed in TUNING_SEEDS:
        data = generate(seed, cfg)
        public = data.catalogue.public()

        t0 = time.perf_counter()
        S = similarity_matrix(public.embeddings)
        t_kernel = time.perf_counter() - t0
        kernel.append({"seconds": t_kernel, "min_eigenvalue": min_eigenvalue(S)})

        r_w = within_category_popularity(data.train_item_counts, public.categories, public.n_categories)
        r_g = global_popularity(data.train_item_counts)
        for (kappa, scen), users in data.users.items():
            res = run_cell(users, public, r_w, r_g, T)
            key = f"kappa={kappa}|{scen}"
            entry = rows.setdefault(key, {m: {"overall": [], "post_shift": []} for m in METHODS} | {"seen_adaptive": []})
            events = switch_events(scen, cfg)
            for m in METHODS:
                cov = res["coverage"][m]
                entry[m]["overall"].append(float(cov[:, BURN_IN:].mean()))
                if events:
                    entry[m]["post_shift"].append(float(np.mean([cov[:, s - 1: s - 1 + POST].mean() for s, _, _ in events])))
            entry["seen_adaptive"].append(float(res["seen"][:, BURN_IN:].mean()))
        print(f"seed {seed} done")

    (out_dir / "check.json").write_text(json.dumps(
        {"seeds": list(TUNING_SEEDS), "k": K, "burn_in": BURN_IN, "post_window": POST,
         "estimators": {"adaptive": PROPOSED_ESTIMATOR.name, "static": STATIC_ESTIMATOR.name},
         "kernel": kernel, "cells": rows}, indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")

    print(f"{'cell':26} " + " ".join(f"{m:>11}" for m in METHODS) + "   (post-shift, else overall)  seen")
    for key, e in rows.items():
        part = "post_shift" if e["popularity"]["post_shift"] else "overall"
        print(f"{key:26} " + " ".join(f"{np.mean(e[m][part]):11.4f}" for m in METHODS) + f"   {part:10}  {np.mean(e['seen_adaptive']):.3f}")
    print("kernel:", kernel[0])
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
