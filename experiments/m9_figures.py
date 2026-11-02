"""Milestone 9: figures and per-scenario tables from the saved M8 records. Runs no simulation.

Per-scenario values are descriptive (D-30): confidence intervals only, outside the Holm family.

    python experiments/m9_figures.py
Writes results/m9_analysis/<run_id>/{tradeoff.png, dpp_effect.png, scenario_ndcg.png, per_scenario.csv}.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptdiv.confirmatory import ALL_SCENARIOS, DRIFT, KAPPAS, pooled  # noqa: E402
from adaptdiv.stats import bootstrap_ci  # noqa: E402

M8_RUN = ROOT / "results" / "m8_main" / "20261102-112410"
KAPPA_LABEL = {0.1: "Focused users (κ = 0.1)", 1.0: "Spread users (κ = 1.0)"}
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COLOURS = {"static": "#2a78d6", "adaptive": "#eb6834", "oracle": "#1baf7a", "popularity": "#898781"}
COMPARISONS = {  # name -> (a, b), value a - b
    "Adaptive - Static, Top-K": ("adaptive_topk", "static_topk"),
    "Adaptive - Static, DPP": ("adaptive_dpp", "static_dpp"),
    "DPP - Top-K, Static": ("static_dpp", "static_topk"),
    "DPP - Top-K, Adaptive": ("adaptive_dpp", "adaptive_topk"),
}


def load(path: Path) -> dict:
    cells = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (float(row["kappa"]), row["scenario"], row["method"], row["phase"], row["metric"])
            cells[key][int(row["seed"])] = float(row["value"])
    return {k: np.array([v[s] for s in sorted(v)]) for k, v in cells.items()}


def write_table(path: Path, table: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kappa", "scenario", "comparison", "phase", "metric", "mean_diff", "ci_low", "ci_high"])
        for kappa in KAPPAS:
            for scen in ALL_SCENARIOS:
                for name, (a, b) in COMPARISONS.items():
                    for phase in ("post", "overall") if scen in DRIFT else ("overall",):
                        for metric in ("ndcg", "eu", "coverage"):
                            diff = table[(kappa, scen, a, phase, metric)] - table[(kappa, scen, b, phase, metric)]
                            w.writerow([kappa, scen, name, phase, metric, diff.mean(), *bootstrap_ci(diff)])


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def tradeoff(table: dict, out: Path, plt) -> None:
    """Post-shift NDCG against coverage, with an arrow from each Top-K to its DPP."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, kappa in zip(axes, KAPPAS):
        style(ax)
        xy = lambda m: (pooled(table, m, "post", "coverage", DRIFT, (kappa,)).mean(),  # noqa: E731
                        pooled(table, m, "post", "ndcg", DRIFT, (kappa,)).mean())
        for profile, name in (("static", "Static"), ("adaptive", "Adaptive"), ("oracle", "Oracle")):
            top, dpp = xy(f"{profile}_topk"), xy(f"{profile}_dpp")
            ax.annotate("", xy=dpp, xytext=top, arrowprops=dict(arrowstyle="-", color=COLOURS[profile], lw=2))
            ax.scatter(*top, s=70, marker="o", color=COLOURS[profile], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.scatter(*dpp, s=70, marker="s", color=COLOURS[profile], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(name, dpp, textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK_2)
        pop = xy("popularity")
        ax.scatter(*pop, s=70, marker="D", color=COLOURS["popularity"], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate("Popularity", pop, textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color=INK_2)
        ax.set_title(KAPPA_LABEL[kappa], fontsize=10, loc="left", color=INK)
        ax.set_xlabel("distinct-category coverage after a shift", fontsize=8, color=INK_2)
        ax.set_ylabel("NDCG@10 after a shift", fontsize=8, color=INK_2)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], marker=m, ls="", color=INK_2, label=t)
               for m, t in (("o", "Top-K"), ("s", "DPP (θ = 0.5)"), ("D", "Popularity"))]
    axes[1].legend(handles=handles, fontsize=8, frameon=False, loc="upper right", labelcolor=INK_2)
    fig.suptitle("Adaptation moves lists up; the DPP moves them right", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def dpp_effect(table: dict, out: Path, plt) -> None:
    """DPP - Top-K on post-shift NDCG, per scenario, with paired 95% CIs across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE, sharey=True)
    width = 0.36
    for ax, kappa in zip(axes, KAPPAS):
        style(ax)
        ax.grid(axis="x", visible=False)
        for shift, profile, name in ((-width / 2, "static", "Static profile"), (width / 2, "adaptive", "Adaptive profile")):
            diffs = [table[(kappa, s, f"{profile}_dpp", "post", "ndcg")] - table[(kappa, s, f"{profile}_topk", "post", "ndcg")] for s in DRIFT]
            means = np.array([d.mean() for d in diffs])
            cis = np.array([bootstrap_ci(d) for d in diffs])
            x = np.arange(len(DRIFT)) + shift
            ax.bar(x, means, width * 0.92, color=COLOURS[profile], label=name)
            ax.errorbar(x, means, yerr=[means - cis[:, 0], cis[:, 1] - means], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(DRIFT)), [s.capitalize() for s in DRIFT], fontsize=8)
        ax.set_title(KAPPA_LABEL[kappa], fontsize=10, loc="left", color=INK)
    axes[0].set_ylabel("change in post-shift NDCG@10 (DPP − Top-K)", fontsize=8, color=INK_2)
    axes[1].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle("The DPP helps a stale profile and costs a current one", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def scenario_ndcg(table: dict, out: Path, plt) -> None:
    """Overall NDCG of the 2x2 methods per scenario. Hatching marks the DPP, for greyscale print."""
    series = (("static_topk", "static", "", "Static Top-K"), ("static_dpp", "static", "///", "Static DPP"),
              ("adaptive_topk", "adaptive", "", "Adaptive Top-K"), ("adaptive_dpp", "adaptive", "///", "Adaptive DPP"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    width = 0.2
    for ax, kappa in zip(axes, KAPPAS):
        style(ax)
        ax.grid(axis="x", visible=False)
        for n, (method, profile, hatch, name) in enumerate(series):
            vals = [table[(kappa, s, method, "overall", "ndcg")].mean() for s in ALL_SCENARIOS]
            ax.bar(np.arange(len(ALL_SCENARIOS)) + (n - 1.5) * width, vals, width * 0.9, color=COLOURS[profile],
                   hatch=hatch, edgecolor=SURFACE, linewidth=0, label=name)
        ax.set_xticks(range(len(ALL_SCENARIOS)), [s.capitalize() for s in ALL_SCENARIOS], fontsize=8)
        ax.set_title(KAPPA_LABEL[kappa], fontsize=10, loc="left", color=INK)
        ax.set_ylabel("NDCG@10, all steps after burn-in", fontsize=8, color=INK_2)
    fig.suptitle("Adaptation pays under drift and costs up to 4.5% without it", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="lower center", fontsize=8, frameon=False, labelcolor=INK_2)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = load(M8_RUN / "records.csv")
    out = ROOT / "results" / "m9_analysis" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_table(out / "per_scenario.csv", table)
    tradeoff(table, out / "tradeoff.png", plt)
    dpp_effect(table, out / "dpp_effect.png", plt)
    scenario_ndcg(table, out / "scenario_ndcg.png", plt)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
