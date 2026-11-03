"""Milestone 10 (D-19): figure from a saved calibration run. Runs no simulation.

    python experiments/calibration_figures.py <run folder>
Writes calibration_tradeoff.png into the run folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from m9_figures import COLOURS, INK, INK_2, KAPPA_LABEL, SURFACE, load, style  # noqa: E402

from adaptdiv.confirmatory import DRIFT, KAPPAS, pooled  # noqa: E402

MARKERS = {"topk": ("o", "Top-K"), "dpp": ("s", "DPP (θ = 0.5)"), "calib": ("^", "Calibrated (λ = 0.8)")}


def tradeoff(table: dict, out: Path, plt) -> None:
    """Post-shift NDCG against coverage; lines join each profile's Top-K to its DPP and calibrated lists."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, kappa in zip(axes, KAPPAS):
        style(ax)
        for profile in ("static", "adaptive"):
            xy = {sel: (pooled(table, f"{profile}_{sel}", "post", "coverage", DRIFT, (kappa,)).mean(),
                        pooled(table, f"{profile}_{sel}", "post", "ndcg", DRIFT, (kappa,)).mean()) for sel in MARKERS}
            for sel in ("dpp", "calib"):
                ax.annotate("", xy=xy[sel], xytext=xy["topk"], arrowprops=dict(arrowstyle="-", color=COLOURS[profile], lw=2))
            for sel, (marker, _) in MARKERS.items():
                ax.scatter(*xy[sel], s=70, marker=marker, color=COLOURS[profile], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{profile.capitalize()} profile", xy["calib"], textcoords="offset points", xytext=(8, 4),
                        fontsize=8, color=INK_2)
        ax.set_title(KAPPA_LABEL[kappa], fontsize=10, loc="left", color=INK)
        ax.set_xlabel("distinct-category coverage after a shift", fontsize=8, color=INK_2)
        ax.set_ylabel("NDCG@10 after a shift", fontsize=8, color=INK_2)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.15)
    handles = [plt.Line2D([], [], marker=m, ls="", color=INK_2, label=t) for m, t in MARKERS.values()]
    fig.legend(handles=handles, ncols=3, loc="lower center", fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle("Calibration and the DPP at the same relevance budget (exploratory seeds 200–219)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(sys.argv[1])
    tradeoff(load(run / "records.csv"), run / "calibration_tradeoff.png", plt)
    print(f"wrote figure to {run}")


if __name__ == "__main__":
    main()
