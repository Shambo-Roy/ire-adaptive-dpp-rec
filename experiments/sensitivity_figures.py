"""Milestone 9: figures from a saved sensitivity run. Runs no simulation.

    python experiments/sensitivity_figures.py <run folder>
Writes frontier.png and interaction.png into the run folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from m9_figures import AXIS, COLOURS, INK, INK_2, KAPPA_LABEL, SURFACE, load, style  # noqa: E402
from sensitivity_study import DEFAULT_ETA, PROFILES, THETAS  # noqa: E402

from adaptdiv.confirmatory import DRIFT, KAPPAS, pooled  # noqa: E402
from adaptdiv.stats import bootstrap_ci  # noqa: E402


def frontier(table: dict, out: Path, plt) -> None:
    """Post-shift NDCG against coverage as theta falls from Top-K, one line per profile."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    thetas = THETAS[DEFAULT_ETA]
    for ax, kappa in zip(axes, KAPPAS):
        style(ax)
        for profile in PROFILES:
            xy = [tuple(pooled(table, f"eta{DEFAULT_ETA}/{profile}_{t}", "post", m, DRIFT, (kappa,)).mean()
                        for m in ("coverage", "ndcg")) for t in thetas]
            ax.plot(*zip(*xy), color=COLOURS[profile], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5,
                    label=f"{profile.capitalize()} profile")
            for t, point in zip(thetas, xy):
                ax.annotate("Top-K" if t == 1.0 else f"θ {t}", point, textcoords="offset points",
                            xytext=(0, -13 if t == 1.0 else 7), ha="center", fontsize=7, color=INK_2)
        ax.set_title(KAPPA_LABEL[kappa], fontsize=10, loc="left", color=INK)
        ax.set_xlabel("distinct-category coverage after a shift", fontsize=8, color=INK_2)
        ax.set_ylabel("NDCG@10 after a shift", fontsize=8, color=INK_2)
        ax.margins(0.08)
    axes[1].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle("Trade-off curves as θ falls from Top-K (exploratory seeds 200–219)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def interaction_values(table: dict, eta: float, theta: float) -> np.ndarray:
    gain = lambda p: (pooled(table, f"eta{eta}/{p}_{theta}", "post", "ndcg", DRIFT)  # noqa: E731
                      - pooled(table, f"eta{eta}/{p}_1.0", "post", "ndcg", DRIFT))
    return gain("adaptive") - gain("static")


def interaction(table: dict, out: Path, plt) -> None:
    """The adaptation x selector interaction with 95% CIs, along each sweep. Below zero is substitution."""
    sweeps = (("θ (noise rate η = 0.05)", [(DEFAULT_ETA, t) for t in sorted(THETAS[DEFAULT_ETA]) if t < 1.0], 1),
              ("click noise rate η (θ = 0.5)", [(e, 0.5) for e in sorted(THETAS)], 0))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (label, grid, axis) in zip(axes, sweeps):
        style(ax)
        x = np.array([g[axis] for g in grid])
        values = [interaction_values(table, *g) for g in grid]
        means = np.array([v.mean() for v in values])
        cis = np.array([bootstrap_ci(v) for v in values])
        ax.errorbar(x, means, yerr=[means - cis[:, 0], cis[:, 1] - means], color=COLOURS["adaptive"], lw=2,
                    marker="o", ms=7, mec=SURFACE, mew=1.5, ecolor=INK, elinewidth=1, capsize=2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(x)
        ax.set_xlabel(label, fontsize=8, color=INK_2)
    axes[0].set_ylabel("interaction on post-shift NDCG@10", fontsize=8, color=INK_2)
    fig.suptitle("Adaptation × selector interaction across the sweeps; below zero is substitution", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(sys.argv[1])
    table = load(run / "records.csv")
    frontier(table, run / "frontier.png", plt)
    interaction(table, run / "interaction.png", plt)
    print(f"wrote figures to {run}")


if __name__ == "__main__":
    main()
