"""Milestone 9, exploratory: sensitivity to the DPP trade-off theta and click noise eta, seeds 200-219.

Static and adaptive profiles, each with Top-K (theta = 1) and the DPP. At the default eta = 0.05
theta takes 0.9, 0.7, 0.5, 0.3; at eta in {0, 0.15, 0.3} only theta = 0.5 runs. eta also changes
the training population, so popularity is re-estimated at each level. The "noisy" scenario keeps
its own eta = 0.3 and runs only at the default. Results are descriptive and change no frozen choice.

    python experiments/sensitivity_study.py --dry-run --out <dir>   # tuning seeds, tiny scale
    python experiments/sensitivity_study.py                         # full scale, about 35 minutes

Writes results/m9_sensitivity/<run_id>/{config.json, records.csv, effects.json, environment.json}.
Method names in records.csv are "eta<eta>/<profile>_<theta>" and "eta<eta>/est:<estimator>".
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

from adaptdiv.confirmatory import DRIFT, KAPPAS, pooled, stack_records  # noqa: E402
from adaptdiv.config import EVALUATION_SEEDS, EXPLORATORY_SEEDS, SCENARIOS, TUNING_SEEDS, PopulationConfig, SimulationConfig  # noqa: E402
from adaptdiv.evaluation import EvalConfig, evaluate  # noqa: E402
from adaptdiv.recommenders import Method  # noqa: E402
from adaptdiv.simulator import generate  # noqa: E402
from adaptdiv.stats import bootstrap_ci  # noqa: E402

DEFAULT_ETA = SimulationConfig().interaction.noise_rate
THETAS = {0.0: (1.0, 0.5), DEFAULT_ETA: (1.0, 0.9, 0.7, 0.5, 0.3), 0.15: (1.0, 0.5), 0.3: (1.0, 0.5)}
PROFILES = ("static", "adaptive")
KAPPA_GROUPS = {"all": KAPPAS, "focused": (0.1,), "spread": (1.0,)}


def run(job: tuple[int, float, SimulationConfig, EvalConfig]) -> dict:
    seed, eta, cfg, ev = job
    scenarios = SCENARIOS if eta == DEFAULT_ETA else tuple(s for s in SCENARIOS if s != "noisy")
    methods = tuple(Method(f"{p}_{t}", p, t) for p in PROFILES for t in THETAS[eta])
    data = generate(seed, replace(cfg, interaction=replace(cfg.interaction, noise_rate=eta)), scenarios)
    return {(k, s, f"eta{eta}/{m}", phase, metric): v for (k, s, m, phase, metric), v in evaluate(data, methods, ev).items()}


def stat(values: np.ndarray) -> dict:
    low, high = bootstrap_ci(values)
    return {"mean": float(values.mean()), "ci_low": low, "ci_high": high}


def effects(table: dict) -> list[dict]:
    """Per eta, theta and kappa group: DPP effects, their interaction, and the cost of diversity."""
    out = []
    for group, kappas in KAPPA_GROUPS.items():
        for eta, thetas in THETAS.items():
            post = lambda p, t, m: pooled(table, f"eta{eta}/{p}_{t}", "post", m, DRIFT, kappas)  # noqa: E731
            stable = lambda p, t, m: pooled(table, f"eta{eta}/{p}_{t}", "overall", m, ("stable",), kappas)  # noqa: E731
            for theta in thetas:
                row = {"kappas": group, "eta": eta, "theta": theta}
                for p in PROFILES:
                    row[p] = {"post_ndcg": stat(post(p, theta, "ndcg")), "post_coverage": stat(post(p, theta, "coverage")),
                              "post_eu": stat(post(p, theta, "eu")), "stable_ndcg": stat(stable(p, theta, "ndcg"))}
                row["adaptation_post_ndcg"] = stat(post("adaptive", theta, "ndcg") - post("static", theta, "ndcg"))
                gap = stable("adaptive", theta, "ndcg") - stable("static", theta, "ndcg")
                row["adaptation_stable_ndcg_relative"] = stat(gap / stable("static", theta, "ndcg"))
                if theta < 1.0:
                    gains = {p: post(p, theta, "ndcg") - post(p, 1.0, "ndcg") for p in PROFILES}
                    row["dpp_post_ndcg"] = {p: stat(g) for p, g in gains.items()}
                    row["interaction"] = stat(gains["adaptive"] - gains["static"])
                    row["dpp_post_coverage"] = {p: stat(post(p, theta, "coverage") - post(p, 1.0, "coverage")) for p in PROFILES}
                    row["dpp_eu_loss_relative"] = {p: stat(1 - post(p, theta, "eu") / post(p, 1.0, "eu")) for p in PROFILES}
                out.append(row)
            for name in ("cumulative", "ewma"):
                est = lambda scen, phase, m: pooled(table, f"eta{eta}/est:{name}", phase, m, scen, kappas)  # noqa: E731
                out.append({"kappas": group, "eta": eta, "estimator": name,
                            "post_tracking": stat(est(DRIFT, "post", "tracking")),
                            "stable_tracking": stat(est(("stable",), "overall", "tracking")),
                            "stable_false_switch": stat(est(("stable",), "overall", "false_switch"))})
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

    out = (args.out or ROOT / "results" / "m9_sensitivity") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(s, eta, cfg, ev) for s in seeds for eta in THETAS]
    t0 = time.perf_counter()
    merged = {s: {} for s in seeds}
    with Pool(PROCESSES) as pool:
        for done, (job, records) in enumerate(zip(jobs, pool.imap(run, jobs)), 1):
            merged[job[0]].update(records)
            print(f"{done}/{len(jobs)} jobs, {time.perf_counter() - t0:.0f}s", flush=True)
    runtime = time.perf_counter() - t0

    per_seed = [merged[s] for s in seeds]
    cfg.save_json(out / "config.json")
    write_records(out / "records.csv", seeds, per_seed)
    (out / "effects.json").write_text(json.dumps(effects(stack_records(per_seed)), indent=2), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
         "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "seeds": seeds, "thetas": {str(k): v for k, v in THETAS.items()}, "processes": PROCESSES,
         "eval_config": vars(ev), "exploratory": True, "runtime_seconds": runtime}, indent=2), encoding="utf-8")
    print(f"runtime {runtime:.0f}s; wrote {out}")


if __name__ == "__main__":
    main()
