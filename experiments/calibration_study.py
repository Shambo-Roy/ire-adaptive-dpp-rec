"""Milestone 10, exploratory (D-19): calibrated re-ranker vs DPP vs Top-K, seeds 200-219.

Static and adaptive profiles, each with Top-K, the DPP (frozen theta) and the calibrated re-ranker
(lambda chosen on tuning seeds by experiments/select_calibration.py). Paired CIs only; no frozen
result changes.

    python experiments/calibration_study.py --lam <lambda> --dry-run --out <dir>   # tuning seeds, tiny scale
    python experiments/calibration_study.py --lam <lambda>                         # full scale

Writes results/m10_calibration/<run_id>/{config.json, records.csv, effects.json, environment.json}.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per seed process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
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

from adaptdiv.confirmatory import ALL_SCENARIOS, DRIFT, KAPPAS, pooled, stack_records  # noqa: E402
from adaptdiv.config import EVALUATION_SEEDS, EXPLORATORY_SEEDS, TUNING_SEEDS, PopulationConfig, SimulationConfig  # noqa: E402
from adaptdiv.evaluation import EvalConfig, evaluate_seed  # noqa: E402
from adaptdiv.recommenders import SELECTED_THETA, Method  # noqa: E402
from adaptdiv.stats import bootstrap_ci  # noqa: E402

PROFILES = ("static", "adaptive")
SELECTORS = ("topk", "dpp", "calib")
KAPPA_GROUPS = {"all": KAPPAS, "focused": (0.1,), "spread": (1.0,)}


def methods(lam: float) -> tuple[Method, ...]:
    out = []
    for p in PROFILES:
        out += [Method(f"{p}_topk", p), Method(f"{p}_dpp", p, SELECTED_THETA), Method(f"{p}_calib", p, calib_lambda=lam)]
    return tuple(out)


def run(job: tuple[int, float, SimulationConfig, EvalConfig]) -> dict:
    seed, lam, cfg, ev = job
    return evaluate_seed(seed, cfg, methods(lam), ev)


def stat(values: np.ndarray) -> dict:
    low, high = bootstrap_ci(values)
    return {"mean": float(values.mean()), "ci_low": low, "ci_high": high}


def effects(table: dict) -> list[dict]:
    """Per kappa group: method means, calibrated - DPP per profile, and each selector's interaction."""
    out = []
    windows = {"post": ("post", DRIFT), "overall": ("overall", ALL_SCENARIOS), "stable": ("overall", ("stable",))}
    for group, kappas in KAPPA_GROUPS.items():
        get = lambda m, w, metric: pooled(table, m, windows[w][0], metric, windows[w][1], kappas)  # noqa: E731
        row = {"kappas": group, "means": {}, "calib_minus_dpp": {}, "calib_minus_topk": {}, "interaction_post_ndcg": {}}
        for p in PROFILES:
            for sel in SELECTORS:
                m = f"{p}_{sel}"
                row["means"][m] = {f"{w}_{metric}": float(get(m, w, metric).mean())
                                   for w in windows for metric in ("ndcg", "eu", "coverage", "miscal")}
            for w in windows:
                for metric in ("ndcg", "eu", "coverage", "miscal"):
                    row["calib_minus_dpp"][f"{p}_{w}_{metric}"] = stat(get(f"{p}_calib", w, metric) - get(f"{p}_dpp", w, metric))
                    row["calib_minus_topk"][f"{p}_{w}_{metric}"] = stat(get(f"{p}_calib", w, metric) - get(f"{p}_topk", w, metric))
        for sel in ("dpp", "calib"):
            gain = lambda p: get(f"{p}_{sel}", "post", "ndcg") - get(f"{p}_topk", "post", "ndcg")  # noqa: E731
            row["interaction_post_ndcg"][sel] = stat(gain("adaptive") - gain("static"))
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, required=True, help="lambda selected on tuning seeds")
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

    out = (args.out or ROOT / "results" / "m10_calibration") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    with Pool(PROCESSES) as pool:
        per_seed = []
        for done, records in enumerate(pool.imap(run, [(s, args.lam, cfg, ev) for s in seeds]), 1):
            per_seed.append(records)
            print(f"{done}/{len(seeds)} seeds, {time.perf_counter() - t0:.0f}s", flush=True)
    runtime = time.perf_counter() - t0

    cfg.save_json(out / "config.json")
    write_records(out / "records.csv", seeds, per_seed)
    (out / "effects.json").write_text(json.dumps(effects(stack_records(per_seed)), indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
         "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "seeds": seeds, "theta": SELECTED_THETA, "lambda": args.lam, "processes": PROCESSES,
         "eval_config": vars(ev), "exploratory": True, "runtime_seconds": runtime}, indent=2), encoding="utf-8")
    print(f"runtime {runtime:.0f}s; wrote {out}")


if __name__ == "__main__":
    main()
