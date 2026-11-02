"""Milestone 8: the confirmatory run on EVALUATION seeds, plus ablations (D-16, D-18, D-30).

The analysis is the frozen one in adaptdiv/confirmatory.py. The ablation comparisons are
descriptive: confidence intervals only, outside the Holm family.

    python experiments/main_experiment.py --dry-run --out <dir>   # tuning seeds, tiny scale
    python experiments/main_experiment.py                         # full scale, about 15 minutes

Writes results/m8_main/<run_id>/{config.json, records.csv, summary.csv, confirmatory.json,
ablations.json, environment.json}. A full run refuses to start if adaptdiv/ differs from HEAD.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per seed process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptdiv.confirmatory import ALL_SCENARIOS, DRIFT, analyse, pooled, stack_records  # noqa: E402
from adaptdiv.config import EVALUATION_SEEDS, TUNING_SEEDS, PopulationConfig, SimulationConfig  # noqa: E402
from adaptdiv.evaluation import EvalConfig, evaluate_seed  # noqa: E402
from adaptdiv.recommenders import SELECTED_THETA, Method, standard_methods  # noqa: E402
from adaptdiv.stats import bootstrap_ci  # noqa: E402

PROCESSES = 5
DRY_USERS, DRY_TRAIN_USERS, DRY_STRIDE = 5, 100, 25
ABLATIONS = {  # base method -> variant
    "adaptive_topk": ("uniform_start_topk", "window_topk"),
    "adaptive_dpp": ("uniform_start_dpp", "window_dpp"),
}


def methods() -> tuple[Method, ...]:
    return standard_methods(SELECTED_THETA) + (
        Method("uniform_start_topk", "uniform_start"),
        Method("uniform_start_dpp", "uniform_start", SELECTED_THETA),
        Method("window_topk", "window"),
        Method("window_dpp", "window", SELECTED_THETA),
    )


def run(job: tuple[int, SimulationConfig, EvalConfig]) -> dict:
    seed, cfg, ev = job
    return evaluate_seed(seed, cfg, methods(), ev)


def write_records(path: Path, seeds: list[int], per_seed: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "kappa", "scenario", "method", "phase", "metric", "value"])
        for seed, records in zip(seeds, per_seed):
            for key, value in records.items():
                w.writerow([seed, *key, repr(value)])


def write_summary(path: Path, table: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kappa", "scenario", "method", "phase", "metric", "mean", "sd", "min", "max"])
        for key, values in table.items():
            w.writerow([*key, values.mean(), values.std(ddof=1), values.min(), values.max()])


def ablations(table: dict) -> list[dict]:
    """Variant - base, drift pooled (post) and all scenarios (overall). Descriptive only."""
    out = []
    for base, variants in ABLATIONS.items():
        for variant in variants:
            for phase, scenarios in (("post", DRIFT), ("overall", ALL_SCENARIOS)):
                for metric in ("ndcg", "eu", "coverage"):
                    a = pooled(table, variant, phase, metric, scenarios)
                    b = pooled(table, base, phase, metric, scenarios)
                    low, high = bootstrap_ci(a - b)
                    out.append({"variant": variant, "base": base, "phase": phase, "metric": metric,
                                "mean_variant": float(a.mean()), "mean_base": float(b.mean()),
                                "mean_diff": float((a - b).mean()), "ci_low": low, "ci_high": high})
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="tuning seeds at tiny scale; never evaluation seeds")
    parser.add_argument("--out", type=Path, help="output folder; required with --dry-run")
    args = parser.parse_args()
    if args.dry_run and args.out is None:
        parser.error("--dry-run needs --out, so a rehearsal cannot land among the results")

    cfg, ev = SimulationConfig(), EvalConfig()
    seeds = list(EVALUATION_SEEDS)
    if args.dry_run:
        seeds = list(TUNING_SEEDS)
        cfg = replace(cfg, population=PopulationConfig(users_per_cell=DRY_USERS, train_users=DRY_TRAIN_USERS))
        ev = replace(ev, stride=DRY_STRIDE)
    elif git("status", "--porcelain", "--", "adaptdiv"):
        sys.exit("adaptdiv/ differs from HEAD: the frozen code must be committed before a full run")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (args.out or ROOT / "results" / "m8_main") / run_id
    out.mkdir(parents=True, exist_ok=False)

    t0 = time.perf_counter()
    with Pool(PROCESSES) as pool:
        per_seed = []
        for done, records in enumerate(pool.imap(run, [(s, cfg, ev) for s in seeds]), 1):
            per_seed.append(records)
            print(f"{done}/{len(seeds)} seeds, {time.perf_counter() - t0:.0f}s", flush=True)
    runtime = time.perf_counter() - t0

    table = stack_records(per_seed)
    cfg.save_json(out / "config.json")
    write_records(out / "records.csv", seeds, per_seed)
    write_summary(out / "summary.csv", table)
    (out / "confirmatory.json").write_text(json.dumps(
        {"seeds": seeds, "dry_run": args.dry_run, **analyse(table)}, indent=2, default=float), encoding="utf-8")
    (out / "ablations.json").write_text(json.dumps(ablations(table), indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
         "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "seeds": seeds, "theta": SELECTED_THETA, "processes": PROCESSES,
         "eval_config": vars(ev), "methods": [vars(m) for m in methods()], "runtime_seconds": runtime},
        indent=2), encoding="utf-8")
    print(f"runtime {runtime:.0f}s; wrote {out}")


if __name__ == "__main__":
    main()
