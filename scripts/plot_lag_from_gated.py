#!/usr/bin/env python3
"""Draw the lag figure from the SAME json the paper's table is verified against.

WHY THIS EXISTS
There is an older plot_lag_result.py that reads lag_result.json, which reports a
different estimator (class-balanced AUROC averaged across three analysis seeds).
Its step-20 value is 0.792 where the paper's table says 0.778. Both numbers are
correct on their own terms, but the paper claims "one estimator throughout", and
shipping a figure and a table that disagree on the same quantity is exactly the
inconsistency a reviewer catches. So the figure is drawn here from
changepoint_lag.json -- the file verify_paper_tables.py checks the table against
-- and it plots the interval the table prints, not an across-seed range.

    python scripts/plot_lag_from_gated.py --out figures/lag.pdf
"""
from __future__ import annotations
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "followup", "results", "fragility", "changepoint_lag.json")
BREAK_AT = 50          # first checkpoint where AUROC differs from step 0
HACK_FROM = 10         # first checkpoint where the flag rate departs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/lag.pdf")
    a = ap.parse_args()

    per = json.load(open(_SRC))["per_step"]
    steps = sorted(int(k) for k in per)
    g = lambda f: [per[str(s)][f] for s in steps]
    auroc, lo, hi = g("auroc"), g("ci_lo"), g("ci_hi")
    flag, acc = g("flag_rate"), g("base_rate")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    # The span over which the policy is already gaming the monitor but the
    # monitor's own discrimination has not moved.
    for ax in (ax1, ax2):
        ax.axvspan(HACK_FROM, BREAK_AT, color="0.90", zorder=0)
        ax.grid(alpha=0.25, linewidth=0.5)

    ax1.plot(steps, flag, "o-", color="#c0392b", lw=1.8, ms=4,
             label="flag rate at frozen threshold (no labels needed)")
    ax1.plot(steps, acc, "s-", color="#2c3e50", lw=1.8, ms=4,
             label="true accuracy (verifier)")
    ax1.set_ylabel("rate")
    # centre right is the one empty region: flag rate is at ~1.0 and accuracy
    # at ~0.15 there, so the box hides no data
    ax1.legend(fontsize=7.5, loc="center right", framealpha=0.9)
    ax1.set_ylim(0, 1.05)

    ax2.plot(steps, auroc, "o-", color="#2471a3", lw=1.8, ms=4,
             label="monitor AUROC, scored on each checkpoint's own rollouts")
    ax2.fill_between(steps, lo, hi, color="#2471a3", alpha=0.18,
                     label="95% prompt-clustered bootstrap CI")
    ax2.axhline(auroc[0], color="0.45", ls=":", lw=1.2)
    ax2.set_ylabel("AUROC")
    ax2.set_xlabel("RLOO step")
    ax2.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

    ax1.annotate("policy already gaming the monitor",
                 xy=((HACK_FROM + BREAK_AT) / 2, 0.94), ha="center",
                 fontsize=7.5, color="0.35")
    ax2.annotate(f"AUROC first departs at {BREAK_AT}",
                 xy=(BREAK_AT, auroc[steps.index(BREAK_AT)]),
                 xytext=(BREAK_AT + 8, auroc[0] + 0.012), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))

    out = os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {a.out} from changepoint_lag.json "
          f"(step 0 AUROC {auroc[0]:.3f}, step {BREAK_AT} {auroc[steps.index(BREAK_AT)]:.3f})")


if __name__ == "__main__":
    main()
