"""Milestone 9, exploratory: separate alpha from the EWMA start, on fresh seeds 200-219.

The M8 uniform-start ablation differed from the proposed estimator in two ways at once. This
crosses them: alpha in {0.05, 0.07} x start in {bias-corrected, uniform}, each with Top-K and DPP.
Results are descriptive and change no frozen choice.

    python experiments/alpha_start_study.py --dry-run --out <dir>   # tuning seeds, tiny scale
    python experiments/alpha_start_study.py                         # full scale, about 11 minutes

Writes results/m9_alpha_start/<run_id>/{config.json, records.csv, effects.json, environment.json}.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per seed process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_experiment import DRY_STRIDE, DRY_TRAIN_USERS, DRY_USERS, PROCESSES, git, write_records  # noqa: E402

from adaptdiv.confirmatory import ALL_SCENARIOS, DRIFT, pooled, stack_records  # noqa: E402
from adaptdiv.config import EVALUATION_SEEDS, EXPLORATORY_SEEDS, TUNING_SEEDS, PopulationConfig, SimulationConfig  # noqa: E402
from adaptdiv.estimators import EstimatorSpec  # noqa: E402
from adaptdiv.evaluation import EvalConfig, evaluate_seed  # noqa: E402
from adaptdiv.recommenders import SELECTED_THETA, Method  # noqa: E402
from adaptdiv.stats import bootstrap_ci  # noqa: E402

ALPHAS = (0.05, 0.07)
STARTS = {"bc": "ewma_bc", "uniform": "ewma"}
SELECTORS = {"topk": 1.0, "dpp": SELECTED_THETA}
PROFILES = {f"{s}_{a}": EstimatorSpec(kind, a) for s, kind in STARTS.items() for a in ALPHAS}


def methods() -> tuple[Method, ...]:
    return tuple(Method(f"{p}_{sel}", p, theta) for p in PROFILES for sel, theta in SELECTORS.items())


def run(job: tuple[int, SimulationConfig, EvalConfig]) -> dict:
    seed, cfg, ev = job
    return evaluate_seed(seed, cfg, methods(), ev, PROFILES)


def effects(table: dict) -> list[dict]:
    """Main effects and interaction per selector: alpha 0.07 - 0.05, uniform - bias-corrected."""
    out = []
    for sel in SELECTORS:
        for phase, scenarios in (("post", DRIFT), ("overall", ALL_SCENARIOS), ("overall", ("stable",))):
            for metric in ("ndcg", "eu", "coverage"):
                cell = {(s, a): pooled(table, f"{s}_{a}_{sel}", phase, metric, scenarios) for s in STARTS for a in ALPHAS}
                alpha_effect = ((cell["bc", 0.07] - cell["bc", 0.05]) + (cell["uniform", 0.07] - cell["uniform", 0.05])) / 2
                start_effect = ((cell["uniform", 0.05] - cell["bc", 0.05]) + (cell["uniform", 0.07] - cell["bc", 0.07])) / 2
                interaction = (cell["uniform", 0.07] - cell["bc", 0.07]) - (cell["uniform", 0.05] - cell["bc", 0.05])
                row = {"selector": sel, "phase": phase, "scenarios": list(scenarios), "metric": metric,
                       "cell_means": {f"{s}_{a}": float(v.mean()) for (s, a), v in cell.items()}}
                for name, diff in (("alpha", alpha_effect), ("start", start_effect), ("interaction", interaction)):
                    low, high = bootstrap_ci(diff)
                    row[name] = {"mean": float(diff.mean()), "ci_low": low, "ci_high": high}
                out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="tuning seeds at tiny scale")
    parser.add_argument("--out", type=Path, help="output folder; required with --dry-run")
    args = parser.parse_args()
    if args.dry_run and args.out is None:
        parser.error("--dry-run needs --out, so a rehearsal cannot land among the results")
    assert not set(EXPLORATORY_SEEDS) & set(EVALUATION_SEEDS) and not set(EXPLORATORY_SEEDS) & set(TUNING_SEEDS)

    cfg, ev, seeds = SimulationConfig(), EvalConfig(), list(EXPLORATORY_SEEDS)
    if args.dry_run:
        seeds = list(TUNING_SEEDS)
        cfg = replace(cfg, population=PopulationConfig(users_per_cell=DRY_USERS, train_users=DRY_TRAIN_USERS))
        ev = replace(ev, stride=DRY_STRIDE)

    out = (args.out or ROOT / "results" / "m9_alpha_start") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    with Pool(PROCESSES) as pool:
        per_seed = []
        for done, records in enumerate(pool.imap(run, [(s, cfg, ev) for s in seeds]), 1):
            per_seed.append(records)
            print(f"{done}/{len(seeds)} seeds, {time.perf_counter() - t0:.0f}s", flush=True)
    runtime = time.perf_counter() - t0

    cfg.save_json(out / "config.json")
    write_records(out / "records.csv", seeds, per_seed)
    (out / "effects.json").write_text(json.dumps(effects(stack_records(per_seed)), indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
         "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
         "seeds": seeds, "theta": SELECTED_THETA, "processes": PROCESSES, "exploratory": True,
         "runtime_seconds": runtime}, indent=2), encoding="utf-8")
    print(f"runtime {runtime:.0f}s; wrote {out}")


if __name__ == "__main__":
    main()
