"""Milestone 7: choose theta on TUNING seeds under the D-29 rule, and rehearse the analysis pipeline.

Rule: the smallest theta in the grid whose EU@10 loss against same-profile Top-K is at most 3%,
for BOTH the static and adaptive profiles, pooled over all scenarios and both kappa levels.

The rehearsal runs the frozen analysis on tuning seeds only to prove the code path works. With
five seeds it has little power and its output is not a result.

    python experiments/select_theta.py             # full default scale, about 10 minutes
Writes results/m7_theta/<run_id>/{config.json, theta.json, rehearsal.json, environment.json}.
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

from adaptdiv.confirmatory import ALL_SCENARIOS, KAPPAS, Names, analyse, stack_records  # noqa: E402
from adaptdiv.config import TUNING_SEEDS, SimulationConfig  # noqa: E402
from adaptdiv.evaluation import EvalConfig, evaluate_seed  # noqa: E402
from adaptdiv.recommenders import Method  # noqa: E402

THETAS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.3)
MAX_LOSS = 0.03
PROFILES = ("static", "adaptive")


def methods() -> tuple[Method, ...]:
    out = [Method("popularity", "none"), Method("static_topk", "static"), Method("adaptive_topk", "adaptive")]
    out += [Method(f"{p}_dpp@{t}", p, t) for p in PROFILES for t in THETAS]
    return tuple(out)


def run(seed: int) -> dict:
    return evaluate_seed(seed, SimulationConfig(), methods(), EvalConfig())


def pooled_all(table, method: str, metric: str) -> np.ndarray:
    return np.mean([table[(k, s, method, "overall", metric)] for k in KAPPAS for s in ALL_SCENARIOS], axis=0)


def select(table) -> tuple[float | None, dict]:
    curve = {}
    for t in THETAS:
        row = {}
        for p in PROFILES:
            eu, top = pooled_all(table, f"{p}_dpp@{t}", "eu"), pooled_all(table, f"{p}_topk", "eu")
            cov, cov_top = pooled_all(table, f"{p}_dpp@{t}", "coverage"), pooled_all(table, f"{p}_topk", "coverage")
            row[p] = {"eu_loss": float(1 - eu.mean() / top.mean()), "coverage": float(cov.mean()), "coverage_topk": float(cov_top.mean())}
        curve[t] = row
    allowed = [t for t in THETAS if all(curve[t][p]["eu_loss"] <= MAX_LOSS for p in PROFILES)]
    return (min(allowed) if allowed else None), curve


def main() -> None:
    seeds = list(TUNING_SEEDS)
    t0 = time.perf_counter()
    with Pool(len(seeds)) as pool:
        per_seed = pool.map(run, seeds)
    runtime = time.perf_counter() - t0
    table = stack_records(per_seed)

    theta, curve = select(table)
    print(f"runtime {runtime:.0f}s; selected theta = {theta}")
    for t, row in curve.items():
        print(t, {p: (round(v['eu_loss'], 4), round(v['coverage'], 3)) for p, v in row.items()})

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / "m7_theta" / run_id
    out.mkdir(parents=True, exist_ok=True)
    SimulationConfig().save_json(out / "config.json")
    (out / "theta.json").write_text(json.dumps(
        {"seeds": seeds, "grid": THETAS, "max_loss": MAX_LOSS, "selected_theta": theta,
         "curve": {str(k): v for k, v in curve.items()}, "runtime_seconds": runtime}, indent=2), encoding="utf-8")
    if theta is not None:
        names = Names(static_dpp=f"static_dpp@{theta}", adaptive_dpp=f"adaptive_dpp@{theta}")
        (out / "rehearsal.json").write_text(json.dumps(
            {"note": "pipeline rehearsal on 5 tuning seeds; NOT a result", **analyse(table, names)}, indent=2, default=float), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
