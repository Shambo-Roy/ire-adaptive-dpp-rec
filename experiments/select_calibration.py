"""Milestone 10 (D-19): choose the calibration trade-off lambda on TUNING seeds, under the D-29 rule.

Rule: the largest (most calibrating) lambda in the grid whose EU@10 loss against same-profile Top-K
is at most 3%, for BOTH the static and adaptive profiles, pooled over all scenarios and both kappa.

    python experiments/select_calibration.py
Writes results/m10_calibration_lambda/<run_id>/{config.json, lambda.json, environment.json}.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per seed process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_experiment import git  # noqa: E402
from select_theta import MAX_LOSS, PROFILES, pooled_all  # noqa: E402

from adaptdiv.config import TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.evaluation import EvalConfig, evaluate_seed  # noqa: E402
from adaptdiv.recommenders import Method  # noqa: E402

LAMBDAS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)


def methods() -> tuple[Method, ...]:
    out = [Method(f"{p}_topk", p) for p in PROFILES]
    out += [Method(f"{p}_calib@{lam}", p, calib_lambda=lam) for p in PROFILES for lam in LAMBDAS]
    return tuple(out)


def run(seed: int) -> dict:
    return evaluate_seed(seed, SimulationConfig(), methods(), EvalConfig())


def select(table) -> tuple[float | None, dict]:
    curve = {}
    for lam in LAMBDAS:
        row = {}
        for p in PROFILES:
            name, top = f"{p}_calib@{lam}", f"{p}_topk"
            row[p] = {"eu_loss": float(1 - pooled_all(table, name, "eu").mean() / pooled_all(table, top, "eu").mean()),
                      "coverage": float(pooled_all(table, name, "coverage").mean()),
                      "coverage_topk": float(pooled_all(table, top, "coverage").mean()),
                      "miscal": float(pooled_all(table, name, "miscal").mean()),
                      "miscal_topk": float(pooled_all(table, top, "miscal").mean())}
        curve[lam] = row
    allowed = [lam for lam in LAMBDAS if all(curve[lam][p]["eu_loss"] <= MAX_LOSS for p in PROFILES)]
    return (max(allowed) if allowed else None), curve


def main() -> None:
    from adaptdiv.confirmatory import stack_records

    seeds = list(TUNING_SEEDS)
    t0 = time.perf_counter()
    with Pool(len(seeds)) as pool:
        per_seed = pool.map(run, seeds)
    runtime = time.perf_counter() - t0
    lam, curve = select(stack_records(per_seed))
    print(f"runtime {runtime:.0f}s; selected lambda = {lam}")
    for k, row in curve.items():
        print(k, {p: {m: round(v, 4) for m, v in vals.items()} for p, vals in row.items()})

    out = ROOT / "results" / "m10_calibration_lambda" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    SimulationConfig().save_json(out / "config.json")
    (out / "lambda.json").write_text(json.dumps(
        {"seeds": seeds, "grid": LAMBDAS, "max_loss": MAX_LOSS, "selected_lambda": lam,
         "curve": {str(k): v for k, v in curve.items()}, "runtime_seconds": runtime}, indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
         "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain")}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
