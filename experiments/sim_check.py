"""Milestone 3 simulator check: scenario behaviour summary, example plots, generation timing.

Uses TUNING seeds only (evaluation seeds are reserved for Milestone 8).

    python experiments/sim_check.py                 # all tuning seeds, default config
    python experiments/sim_check.py --seeds 0 1     # subset

Writes results/m3_sim_check/<run_id>/{config.json, summary.json, environment.json, trajectories.png}.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptdiv.config import SCENARIOS, TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.simulator import generate  # noqa: E402

WINDOW = 20  # interactions per window when comparing empirical category frequencies with z_t


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(axis=-1)


def windowed_tracking_error(users, C: int) -> list[float]:
    """Mean over users of TV(empirical category freq in window, mean observed-category dist in window)."""
    T = users[0].truth.z.shape[0]
    out = []
    for start in range(0, T, WINDOW):
        errs = []
        for u in users:
            gt = u.truth
            sl = slice(start, start + WINDOW)
            emp = np.bincount(gt.item_categories[sl], minlength=C) / WINDOW
            target = ((1 - gt.noise_rate) * gt.z[sl] + gt.noise_rate / C).mean(axis=0)
            errs.append(tv(emp, target))
        out.append(float(np.mean(errs)))
    return out


def summarise(data) -> dict:
    C = data.catalogue.n_categories
    cat = data.catalogue
    S = cat.embeddings @ cat.embeddings.T
    same = cat.categories[:, None] == cat.categories[None, :]
    off = ~np.eye(cat.n_items, dtype=bool)
    res = {
        "cos_within": float(S[same & off].mean()),
        "cos_across": float(S[~same].mean()),
        "train_items_never_seen": int((data.train_item_counts == 0).sum()),
        "cells": {},
    }
    for (kappa, scen), users in data.users.items():
        gt0 = [u.truth for u in users]
        res["cells"][f"kappa={kappa}|{scen}"] = {
            "mean_shift_tv": float(np.mean([tv(g.params.z_a, g.params.z_b) for g in gt0])),
            "mean_max_interest": float(np.mean([g.params.z_a.max() for g in gt0])),
            "noise_rate_observed": float(np.mean([g.noise_flags.mean() for g in gt0])),
            "repeat_fraction": float(np.mean([1 - len(np.unique(u.observed.items)) / len(u.observed) for u in users])),
            "window_tracking_tv": windowed_tracking_error(users, C),
        }
    return res


def plot_examples(data, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = data.catalogue.n_categories
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(4 * len(SCENARIOS), 6), sharey=True, sharex=True)
    for row, kappa in enumerate(data.config.population.kappas):
        for col, scen in enumerate(SCENARIOS):
            ax = axes[row, col]
            gt = data.users[(kappa, scen)][0].truth
            top_a, top_b = int(np.argmax(gt.params.z_a)), int(np.argmax(gt.params.z_b))
            t = np.arange(1, gt.z.shape[0] + 1)
            onehot = np.eye(C)[gt.item_categories]
            roll = np.array([onehot[max(0, k - WINDOW + 1): k + 1].mean(axis=0) for k in range(len(t))])
            for c, colour, label in ((top_a, "C0", "top cat. of z_A"), (top_b, "C3", "top cat. of z_B")):
                ax.plot(t, gt.z[:, c], color=colour, lw=2, label=f"true z_t ({label})")
                ax.plot(t, roll[:, c], color=colour, lw=1, ls="--", alpha=0.7, label=f"observed freq, last {WINDOW}")
            ax.set_title(f"{scen}, kappa={kappa}", fontsize=9)
            if row == 1:
                ax.set_xlabel("interaction t")
    axes[0, 0].set_ylabel("interest / frequency")
    axes[1, 0].set_ylabel("interest / frequency")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Example user per scenario (seed %d, user 0): true interest vs observed category frequency" % data.seed)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(TUNING_SEEDS))
    args = ap.parse_args()
    if any(s >= 100 for s in args.seeds):
        raise SystemExit("evaluation seeds (>=100) are reserved for Milestone 8")

    cfg = SimulationConfig()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / "m3_sim_check" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out / "config.json")

    per_seed, timings = {}, {}
    for seed in args.seeds:
        t0 = time.perf_counter()
        data = generate(seed, cfg)
        timings[seed] = time.perf_counter() - t0
        per_seed[seed] = summarise(data)
        if seed == args.seeds[0]:
            plot_examples(data, out / "trajectories.png")
        print(f"seed {seed}: generated in {timings[seed]:.2f}s")

    n_inter = len(cfg.population.kappas) * len(SCENARIOS) * cfg.population.users_per_cell * cfg.scenario.horizon
    summary = {
        "seeds": args.seeds,
        "interactions_per_seed": n_inter,
        "train_interactions_per_seed": cfg.population.train_users * cfg.population.train_horizon,
        "generation_seconds": timings,
        "per_seed": per_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
