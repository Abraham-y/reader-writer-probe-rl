#!/usr/bin/env python3
"""The lag, recomputed separately on the prompts the reward probe was and was
not fit on.

WHY THIS EXISTS
Reviewer xAvm (InterpScience 2026, concern 3): 248 of the 406 evaluation prompts
were among the reward probe's 300 fitting prompts, so the lag ladder is not
prompt-disjoint from the fit. Fresh answers at every checkpoint do not remove
prompt overlap, and the paper's claim that overlap "could shift the level but not
the timing" was asserted, not measured. This measures it: every statistic in the
lag table, computed on the 158 unseen prompts and on the 248 seen ones, with the
same frozen threshold and the same prompt-clustered paired bootstrap as
changepoint_lag.py.

The threshold is the one the paper uses -- the median probe score over ALL 406
prompts at step 0 -- so the flag rates here are directly comparable to the table.
Per-subset thresholds would give each subset a 0.5 flag rate at step 0 and hide
any level difference, which is part of what we want to see.

    python followup/experiments/fragility/phase0_replicate/overlap_split.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from changepoint_lag import _ROOT, auroc, load_checkpoint  # noqa: E402

FIT = os.path.join(_ROOT, "extension", "cache", "probe_cache_temp1",
                   "C_outcome_temp1_l16_pre_answer.meta.json")
STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def paired(data, idx, prompts, a, b, stat, n_boot, rng):
    """Prompt-clustered paired bootstrap of stat(b) - stat(a); returns delta, lo, hi, p."""
    d = np.empty(n_boot)
    for i in range(n_boot):
        dr = rng.choice(prompts, size=len(prompts), replace=True)
        ia = np.concatenate([idx[a][q] for q in dr])
        ib = np.concatenate([idx[b][q] for q in dr])
        d[i] = stat(data[b], ib) - stat(data[a], ia)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d < 0).mean(), (d > 0).mean())
    return float(d.mean()), float(lo), float(hi), float(p)


def compute(n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> dict:
    """Every statistic in the lag table, per subset. Importable, so the camera-ready
    gate recomputes the paper's Table 5 rather than trusting the saved json."""
    data = {s: load_checkpoint("phase0_harvest_runA", s, 16) for s in STEPS}
    fitted = {r["prompt_idx"] for r in json.load(open(FIT))}
    everyone = np.unique(data[0][2])
    thr = float(np.quantile(data[0][0], 0.5))   # the paper's threshold, all 406

    subsets = {
        "unseen": np.array([p for p in everyone if p not in fitted]),
        "seen": np.array([p for p in everyone if p in fitted]),
        "all": everyone,
    }
    auc = lambda d, sel: auroc(d[1][sel], d[0][sel])
    flag = lambda d, sel: float((d[0][sel] >= thr).mean())
    say = print if verbose else (lambda *a, **k: None)

    out = {"threshold": thr, "n_boot": n_boot, "subsets": {}}
    for name, prompts in subsets.items():
        keep = {s: np.isin(data[s][2], prompts) for s in STEPS}
        sub = {s: (data[s][0][keep[s]], data[s][1][keep[s]], data[s][2][keep[s]])
               for s in STEPS}
        idx = {s: {p: np.flatnonzero(sub[s][2] == p) for p in prompts} for s in STEPS}
        rng = np.random.default_rng(seed)
        rows = {}
        say(f"\n{name}: {len(prompts)} prompts  (threshold {thr:.4f}, frozen on all 406)")
        say(f"  {'step':>4} {'n':>5} {'acc':>6} {'AUROC':>6}  {'dAUROC vs 0':>26}"
            f"  {'flag':>6}  {'dflag vs 0':>26}")
        for s in STEPS:
            sc, yy, _ = sub[s]
            r = {"n": int(len(yy)), "acc_last_block": float(yy.mean()),
                 "auroc": auc(sub[s], np.arange(len(yy))),
                 "flag_rate": flag(sub[s], np.arange(len(yy)))}
            if s:
                r["d_auroc"] = paired(sub, idx, prompts, 0, s, auc, n_boot, rng)
                r["d_flag"] = paired(sub, idx, prompts, 0, s, flag, n_boot, rng)
            rows[s] = r
            fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}] p={t[3]:.3f}"
            say(f"  {s:>4} {r['n']:>5} {r['acc_last_block']:>6.3f} {r['auroc']:>6.3f}  "
                f"{fmt(r['d_auroc']) if s else '':>26}  {r['flag_rate']:>6.3f}  "
                f"{fmt(r['d_flag']) if s else '':>26}")
        first_auc = next((s for s in STEPS[1:] if rows[s]["d_auroc"][3] < 0.05), None)
        first_flag = next((s for s in STEPS[1:] if rows[s]["d_flag"][3] < 0.05), None)
        say(f"  first step with p<0.05:  AUROC {first_auc}   flag rate {first_flag}")
        out["subsets"][name] = {"n_prompts": int(len(prompts)), "per_step": rows,
                                "first_auroc_move": first_auc, "first_flag_move": first_flag}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="followup/results/fragility/overlap_split.json")
    a = ap.parse_args()
    out = compute(a.n_boot, a.seed)
    path = os.path.join(_ROOT, a.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
