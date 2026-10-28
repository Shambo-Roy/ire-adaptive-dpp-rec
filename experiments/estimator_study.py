"""Milestone 4 estimator study (exploratory; TUNING seeds only).

1. Tune EWMA alpha and window W on the pooled scenario/kappa mix (D-25), objective =
   mean tracking error TV(zhat_t, z_t) over all steps, users, scenarios, kappa and seeds.
2. Compare cumulative mean, EWMA(alpha*), window(W*) and the window matched in memory to
   alpha* per scenario: post-shift error, crossover lag, false-switch rate.

Confirmatory tests of H1-H3 happen only at M8 on evaluation seeds; nothing here is a
hypothesis test.

    python experiments/estimator_study.py
Writes results/m4_estimators/<run_id>/{config.json, study.json, environment.json, *.png}.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptdiv.config import SCENARIOS, TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.estimation_metrics import crossover_lag, false_switch_rate, noise_floor, tracking_error  # noqa: E402
from adaptdiv.estimators import EstimatorSpec, categories_of, matched_window  # noqa: E402
from adaptdiv.simulator import generate  # noqa: E402

ALPHAS = (0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)
WINDOWS = (3, 5, 7, 10, 15, 20, 30, 40, 60, 100)
BURN_IN = 20       # steps excluded from stable-phase error and false-switch rate
POST = 50          # length of post-shift windows (steps)
RUN = 5            # consecutive steps required by the crossover lag


def load_cells(seed: int, cfg: SimulationConfig) -> dict:
    """Observed category sequences (estimator input) and ground truth (evaluation only) per cell."""
    data = generate(seed, cfg)
    item_cats = data.catalogue.public().categories
    cells = {}
    for (kappa, scen), users in data.users.items():
        cells[(kappa, scen)] = {
            "cats": np.stack([categories_of(u.observed.items, item_cats) for u in users]),  # observable
            "z": np.stack([u.truth.z for u in users]),                                       # truth
            "z_a": np.stack([u.truth.params.z_a for u in users]),
            "z_b": np.stack([u.truth.params.z_b for u in users]),
            "eta": users[0].truth.noise_rate,
        }
    return cells


def switch_events(scen: str, cfg: SimulationConfig) -> list[tuple[int, str, str]]:
    """(start step, old, new) for each interest shift in a scenario."""
    sc = cfg.scenario
    if scen == "abrupt":
        return [(sc.abrupt_switch, "z_a", "z_b")]
    if scen == "gradual":  # truth is equidistant at the ramp midpoint
        return [(sc.gradual_start + sc.gradual_width // 2, "z_a", "z_b")]
    if scen == "recurrent":
        t1, t2 = sc.recurrent_switches
        return [(t1, "z_a", "z_b"), (t2, "z_b", "z_a")]
    return []


def evaluate(spec: EstimatorSpec, cell: dict, scen: str, cfg: SimulationConfig) -> dict:
    C = cell["z"].shape[-1]
    zhat = spec(cell["cats"], C)
    err = tracking_error(zhat, cell["z"])  # (U, T); column k = step t = k+1
    out = {
        "mean_error": float(err.mean()),
        "mean_error_after_burn_in": float(err[:, BURN_IN:].mean()),
    }
    events = switch_events(scen, cfg)
    if events:
        lags, cens, post = [], [], []
        for start, old, new in events:
            lag, c = crossover_lag(zhat, cell[old], cell[new], start, RUN)
            lags.append(lag)
            cens.append(c)
            post.append(err[:, start - 1: start - 1 + POST].mean(axis=1))
        out["crossover_lag_mean"] = float(np.mean(lags))
        out["crossover_lag_median"] = float(np.median(lags))
        out["crossover_censored_frac"] = float(np.mean(cens))
        out["post_shift_error"] = float(np.mean(post))
        first = events[0][0]
        out["pre_shift_error"] = float(err[:, BURN_IN: first - 1].mean())
    else:
        out["false_switch_rate"] = float(false_switch_rate(zhat, cell["z"], BURN_IN).mean())
    out["noise_floor"] = float(noise_floor(cell["z"], cell["eta"]).mean())
    return out


def error_curve(spec: EstimatorSpec, cell: dict) -> np.ndarray:
    zhat = spec(cell["cats"], cell["z"].shape[-1])
    return tracking_error(zhat, cell["z"]).mean(axis=0)


def main() -> None:
    cfg = SimulationConfig()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / "m4_estimators" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out / "config.json")

    seeds = list(TUNING_SEEDS)
    all_cells = {s: load_cells(s, cfg) for s in seeds}
    keys = list(all_cells[seeds[0]].keys())

    # ---- 1. tuning on the pooled mix
    grid = [EstimatorSpec("cumulative")] + [EstimatorSpec("ewma", a) for a in ALPHAS] + [EstimatorSpec("window", w) for w in WINDOWS]
    pooled = {}
    for spec in grid:
        pooled[spec.name] = float(np.mean([
            tracking_error(spec(all_cells[s][k]["cats"], 10), all_cells[s][k]["z"]).mean()
            for s in seeds for k in keys
        ]))
        print(f"{spec.name:16s} pooled mean TV = {pooled[spec.name]:.4f}")
    best_alpha = min(ALPHAS, key=lambda a: pooled[EstimatorSpec("ewma", a).name])
    best_w = min(WINDOWS, key=lambda w: pooled[EstimatorSpec("window", w).name])
    w_match = matched_window(best_alpha)
    print(f"selected alpha* = {best_alpha}, W* = {best_w}, matched window for alpha* = {w_match}")

    # ---- 2. per-scenario comparison of the selected settings
    compare = [EstimatorSpec("cumulative"), EstimatorSpec("ewma", best_alpha), EstimatorSpec("window", best_w)]
    if w_match != best_w:
        compare.append(EstimatorSpec("window", w_match))
    per_cell = {}
    for spec in compare + [EstimatorSpec("ewma", a) for a in ALPHAS]:
        for k in keys:
            vals = [evaluate(spec, all_cells[s][k], k[1], cfg) for s in seeds]
            per_seed = {m: [v[m] for v in vals] for m in vals[0]}
            per_cell[f"{spec.name}|kappa={k[0]}|{k[1]}"] = {
                m: {"mean": float(np.mean(x)), "sd_over_seeds": float(np.std(x, ddof=1)), "per_seed": x}
                for m, x in per_seed.items()
            }

    study = {
        "seeds": seeds,
        "alphas": ALPHAS,
        "windows": WINDOWS,
        "burn_in": BURN_IN,
        "post_window": POST,
        "crossover_run": RUN,
        "pooled_mean_tv": pooled,
        "selected": {"alpha": best_alpha, "window": best_w, "matched_window_for_alpha": w_match},
        "per_cell": per_cell,
    }
    (out / "study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")

    plot(out, all_cells, seeds, keys, compare, pooled, cfg)
    print(f"wrote {out}")


def plot(out: Path, all_cells, seeds, keys, compare, pooled, cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # tuning curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ALPHAS, [pooled[EstimatorSpec("ewma", a).name] for a in ALPHAS], "o-")
    axes[0].axhline(pooled["cumulative"], color="grey", ls="--", label="cumulative mean")
    axes[0].set_xscale("log"); axes[0].set_xlabel("EWMA alpha"); axes[0].set_ylabel("pooled mean TV(zhat_t, z_t)")
    axes[0].legend()
    axes[1].plot(WINDOWS, [pooled[EstimatorSpec("window", w).name] for w in WINDOWS], "o-", color="C2")
    axes[1].axhline(pooled["cumulative"], color="grey", ls="--")
    axes[1].set_xscale("log"); axes[1].set_xlabel("window W")
    fig.suptitle("Tuning on pooled scenario mix (tuning seeds %s)" % seeds)
    fig.tight_layout(); fig.savefig(out / "tuning_curves.png", dpi=110); plt.close(fig)

    # error over time
    kappas = cfg.population.kappas
    fig, axes = plt.subplots(len(kappas), len(SCENARIOS), figsize=(4 * len(SCENARIOS), 3.2 * len(kappas)), sharey="row", sharex=True)
    t = np.arange(1, cfg.scenario.horizon + 1)
    for r, kappa in enumerate(kappas):
        for c, scen in enumerate(SCENARIOS):
            ax = axes[r, c]
            for i, spec in enumerate(compare):
                curve = np.mean([error_curve(spec, all_cells[s][(kappa, scen)]) for s in seeds], axis=0)
                ax.plot(t, curve, lw=1.4, color=f"C{i}", label=spec.name)
            floor = np.mean([noise_floor(all_cells[s][(kappa, scen)]["z"], all_cells[s][(kappa, scen)]["eta"]).mean(axis=0) for s in seeds], axis=0)
            ax.plot(t, floor, color="k", ls=":", lw=1, label="noise floor")
            ax.set_title(f"{scen}, kappa={kappa}", fontsize=9)
            if r == len(kappas) - 1:
                ax.set_xlabel("interaction t")
        axes[r, 0].set_ylabel("mean TV(zhat_t, z_t)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Estimation error over time (mean over users and tuning seeds)")
    fig.tight_layout(); fig.savefig(out / "error_over_time.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
