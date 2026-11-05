#!/usr/bin/env python3
"""Draw the reward ladder: each reward's held-out AUROC against the accuracy of
the policy trained on it (Figure 2).

Every value is recomputed here by the functions the gates use, so the figure
cannot drift from Table 2 without a gate failing: the AUROCs by
verify_reward_ladder.arm_aurocs, the accuracies and paired intervals by
verify_residual_arms (the same 10,000-draw, seed-0 bootstrap as the table).

Each reward is placed at its AUROC on answers cut as in training: RL stopped
sampling at the first `</answer>` (filled). Each is also drawn at its AUROC
on evaluation answers, which run on past it (hollow).

    python scripts/plot_ladder_from_gated.py --out figures/ladder.pdf
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
# NeurIPS asks for Type 1 or TrueType fonts; matplotlib's default is Type 3
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "extension", "probe"))
from verify_reward_ladder import arm_aurocs  # noqa: E402
import verify_residual_arms as arms_gate  # noqa: E402

EVALS = {"A": "eval_armA_residual_step100.json", "B": "eval_armB_surface_step100.json",
         "R": "eval_armRaw_step100.json", "C": "eval_runB_armprotocol_step100.json"}
LABEL = {"A": "arm A: residualised probe", "C": "arm C (runB): other recipe",
         "B": "arm B: text features only", "R": "arm R: raw probe"}
COLOR = {"A": "#2471a3", "C": "0.6", "B": "#c0392b", "R": "#7d3c98"}


def accuracies():
    """First-block accuracy of each arm and the paired CI of its difference
    from arm A, exactly as Table 2 prints them."""
    kp = arms_gate.clean_prompts()
    per = {k: arms_gate.load_arm(f, kp) for k, f in EVALS.items()}
    if any(not v for v in per.values()):
        raise SystemExit("an arm's rollout file is missing; fetch the artifacts first")
    acc = {k: float(np.mean(np.concatenate(list(v.values())))) for k, v in per.items()}
    ci = {}
    for k in "BCR":
        _, lo, hi = arms_gate.paired_bootstrap(per["A"], per[k], 10000, 0)
        ci[k] = (-hi / 100, -lo / 100)        # (arm k - arm A), as a fraction
    return acc, ci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/ladder.pdf")
    a = ap.parse_args()

    trained = arm_aurocs("C_outcome", cut=True)
    evalt = arm_aurocs("C_outcome", cut=False)
    acc, ci = accuracies()

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    on = sorted("ABR", key=trained.get)      # only A, B and R share a recipe
    ax.plot([trained[k] for k in on], [acc[k] for k in on], "-", color="0.55", lw=1.2, zorder=1)
    for k in "AB":                            # evaluation-text AUROC, hollow, with an arrow
        ax.scatter(evalt[k], acc[k], s=60, facecolors="none", edgecolors=COLOR[k], linewidths=1.2, zorder=3)
        ax.annotate("", xy=(trained[k], acc[k]), xytext=(evalt[k], acc[k]),
                    arrowprops=dict(arrowstyle="->", color=COLOR[k], lw=0.9, shrinkA=4, shrinkB=5), zorder=2)
    for k in sorted(trained, key=trained.get):
        if k in ci:
            lo, hi = acc["A"] + ci[k][0], acc["A"] + ci[k][1]
            ax.errorbar(trained[k], acc[k], yerr=[[acc[k] - max(lo, 0)], [hi - acc[k]]],
                        fmt="none", ecolor="0.35", elinewidth=1, capsize=3, zorder=2)
        ax.scatter(trained[k], acc[k], s=70, zorder=4, color=COLOR[k])
        dx, dy, ha, va = {"A": (0.0, 0.016, "center", "bottom"),
                          "C": (-0.004, 0.026, "right", "bottom"),
                          "R": (0.0, 0.026, "center", "bottom"),
                          "B": (0.004, -0.016, "left", "top")}[k]
        ax.annotate(f"{LABEL[k]}\n{trained[k]:.3f} → {acc[k]:.3f}",
                    xy=(trained[k], acc[k]), xytext=(trained[k] + dx, acc[k] + dy),
                    fontsize=7.5, ha=ha, va=va)

    ax.set_xlabel("held-out AUROC of the reward on answers cut as in training\n"
                  "(hollow: on evaluation answers; labels: AUROC → accuracy)")
    ax.set_ylabel("accuracy of the policy trained on it")
    ax.set_xlim(0.815, 1.0); ax.set_ylim(-0.065, 0.23)
    ax.grid(alpha=0.25, linewidth=0.5)

    out = os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})  # byte-reproducible
    print(f"wrote {a.out}: " + ", ".join(f"{k} {trained[k]:.3f} (eval {evalt[k]:.3f}) -> {acc[k]:.4f}"
                                          for k in sorted(trained, key=trained.get)))


if __name__ == "__main__":
    main()
