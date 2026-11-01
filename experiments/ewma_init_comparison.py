"""D-27 evidence: EWMA with a uniform start vs bias-corrected EWMA (exploratory; TUNING seeds only).

Reuses the Milestone 4 study's data loading and metrics, so the two EWMA versions are
compared on exactly the same sequences and objective as results/m4_estimators/.

    python experiments/ewma_init_comparison.py
Writes results/m4_ewma_init/<run_id>/{config.json, comparison.json, environment.json, early_error.png}.
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
sys.path.insert(0, str(ROOT / "experiments"))

from adaptdiv.config import SCENARIOS, TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.estimation_metrics import tracking_error  # noqa: E402
from adaptdiv.estimators import EstimatorSpec  # noqa: E402
from estimator_study import ALPHAS, error_curve, evaluate, load_cells  # noqa: E402

KINDS = ("ewma", "ewma_bc")


def main() -> None:
    cfg = SimulationConfig()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / "m4_ewma_init" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out / "config.json")

    seeds = list(TUNING_SEEDS)
    cells = {s: load_cells(s, cfg) for s in seeds}
    keys = list(cells[seeds[0]].keys())

    pooled = {
        kind: {
            a: float(np.mean([
                tracking_error(EstimatorSpec(kind, a)(cells[s][k]["cats"], 10), cells[s][k]["z"]).mean()
                for s in seeds for k in keys
            ]))
            for a in ALPHAS
        }
        for kind in KINDS
    }
    best = {kind: min(ALPHAS, key=lambda a: pooled[kind][a]) for kind in KINDS}
    for a in ALPHAS:
        print(f"alpha={a:<5} ewma={pooled['ewma'][a]:.4f}  ewma_bc={pooled['ewma_bc'][a]:.4f}")
    print("selected:", best)

    per_cell = {}
    for kind in KINDS:
        for a in sorted(set(best.values())):
            spec = EstimatorSpec(kind, a)
            for k in keys:
                vals = [evaluate(spec, cells[s][k], k[1], cfg) for s in seeds]
                per_cell[f"{spec.name}|kappa={k[0]}|{k[1]}"] = {
                    m: {"mean": float(np.mean([v[m] for v in vals])),
                        "sd_over_seeds": float(np.std([v[m] for v in vals], ddof=1)),
                        "per_seed": [v[m] for v in vals]}
                    for m in vals[0]
                }

    (out / "comparison.json").write_text(json.dumps(
        {"seeds": seeds, "alphas": ALPHAS, "pooled_mean_tv": pooled, "selected": best, "per_cell": per_cell}, indent=2),
        encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")
    plot(out, cells, seeds, best, cfg)
    print(f"wrote {out}")


def plot(out: Path, cells, seeds, best, cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kappas = cfg.population.kappas
    fig, axes = plt.subplots(1, len(kappas), figsize=(5 * len(kappas), 3.6))
    t = np.arange(1, cfg.scenario.horizon + 1)
    for ax, kappa in zip(axes, kappas):
        specs = [EstimatorSpec("cumulative"), EstimatorSpec("ewma", best["ewma"]), EstimatorSpec("ewma_bc", best["ewma_bc"])]
        for i, spec in enumerate(specs):
            curve = np.mean([error_curve(spec, cells[s][(kappa, "stable")]) for s in seeds], axis=0)
            ax.plot(t[:80], curve[:80], color=f"C{i}", label=spec.name)
        ax.set_title(f"stable scenario, kappa={kappa}")
        ax.set_xlabel("interaction t")
    axes[0].set_ylabel("mean TV(zhat_t, z_t)")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(out / "early_error.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
