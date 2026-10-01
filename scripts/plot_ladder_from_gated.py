#!/usr/bin/env python3
"""Draw the reward ladder: read-only AUROC against the accuracy of the policy
trained on that reward, for the three arms scored on one population.

This is contribution 1's evidence and it lived only in a three-row table. The
AUROCs come from the same functions verify_reward_ladder.py gates, so the figure
cannot drift from the table without the gate failing first. The accuracies and
their intervals are the values printed in tab:arms, which verify_residual_arms.py
recomputes; they are restated here with their source rather than re-derived, so
that a change there shows up as a mismatch here.

    python scripts/plot_ladder_from_gated.py --out figures/ladder.pdf
"""
from __future__ import annotations
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from verify_reward_ladder import arm_c, ARMS  # noqa: E402

# first-block accuracy of the policy trained on each reward, all three under the
# arm protocol, and the paired prompt-clustered CI of its difference from arm C
# (tab:arms; verify_residual_arms.py fails if these drift from the rollouts)
ACC = {"A": 0.1678, "C": 0.0782, "B": 0.0000}
# paired CI of each arm's difference from arm A, the controlled reference
DIFF_CI = {"B": (-0.1897, -0.1466), "C": (-0.1081, -0.0717)}
LABEL = {"A": "arm A: surface-residualised probe",
         "C": "arm C: the raw reward probe (runB),\nnot on the arms' recipe",
         "B": "arm B: surface features only"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/ladder.pdf")
    a = ap.parse_args()

    auroc = {
        "A": json.load(open(os.path.join(ARMS, "probe_surface_residual_l16.pkl.meta.json")))["report"]["auroc_residual_heldout"],
        "B": json.load(open(os.path.join(ARMS, "probe_surface_only.pkl.meta.json")))["report"]["auroc_heldout"],
        "C": arm_c()[0],
    }
    order = sorted(auroc, key=auroc.get)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    # only A and B share a recipe, so only they are joined
    ax.plot([auroc["A"], auroc["B"]], [ACC["A"], ACC["B"]], "-", color="0.55", lw=1.2, zorder=1)
    for k in order:
        lo = ACC["A"] + DIFF_CI[k][0] if k in DIFF_CI else None
        hi = ACC["A"] + DIFF_CI[k][1] if k in DIFF_CI else None
        if lo is not None:
            ax.errorbar(auroc[k], ACC[k], yerr=[[ACC[k] - max(lo, 0)], [hi - ACC[k]]],
                        fmt="none", ecolor="0.35", elinewidth=1, capsize=3, zorder=2)
        ax.scatter(auroc[k], ACC[k], s=70, zorder=3,
                   color={"A": "#2471a3", "C": "0.6", "B": "#c0392b"}[k])
        # A and C label up-right into open space; B labels down-left, under the
        # trend line, because above it the text lands on the line itself
        dx, dy, ha, va = {"A": (0.004, 0.012, "left", "bottom"),
                          "C": (-0.004, 0.022, "right", "bottom"),
                          "B": (-0.003, -0.014, "right", "top")}[k]
        ax.annotate(f"{LABEL[k]}\nAUROC {auroc[k]:.3f}, accuracy {ACC[k]:.3f}",
                    xy=(auroc[k], ACC[k]), xytext=(auroc[k] + dx, ACC[k] + dy),
                    fontsize=7.5, ha=ha, va=va)

    ax.set_xlabel("read-only AUROC of the reward")   # population and split are in the caption
    ax.set_ylabel("accuracy of the policy trained on it")
    ax.set_xlim(0.815, 0.965); ax.set_ylim(-0.065, 0.23)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.text(0.98, 0.95, "better reader, worse policy", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="0.3")

    out = os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {a.out}: " + ", ".join(f"{k} {auroc[k]:.3f}->{ACC[k]:.3f}" for k in order))


if __name__ == "__main__":
    main()
