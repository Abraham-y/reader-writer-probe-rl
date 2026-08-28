"""Recompute every cell of the paper's two lag tables from raw data.

WHY THIS EXISTS
An edit pass once overwrote the probe's lag table with the judge's, leaving the
judge series sitting under the probe's caption -- so the claim in the title was
supported in the built PDF by nothing but a figure. Two independent reviewers
found it; no gate did, because every gate checked that the ANALYSIS reproduced
and none checked that the PAPER matched the analysis. This closes that gap: it
parses the tables out of writeup_judge.tex and recomputes each number from the
cached activations and the judge score files.

It fails loudly if a table is swapped, a column is stale, or a row is retyped.

    python scripts/verify_paper_tables.py [--tex writeup_judge.tex]
"""
from __future__ import annotations

import argparse, glob, json, os, pickle, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBE = os.path.join(ROOT, "extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl")
ACTS = os.path.join(ROOT, "followup/acts/phase0_harvest_runA")
JUDGE = os.path.join(ROOT, "followup/results/fragility/judge_lag")
STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
TOL = {"auroc": 0.0015, "prec": 0.0015, "lr": 0.02, "flag": 0.0015,
       # CIs are re-bootstrapped at lower resolution than the published 2,000
       # draws, so they get a Monte-Carlo tolerance rather than an exact match.
       "flag_lo": 0.02, "flag_hi": 0.02}
N_BOOT_CHECK = 500


def stats(y, s, thr, pid=None):
    p = s >= thr
    tpr, fpr = p[y == 1].mean(), p[y == 0].mean()
    out = {"auroc": roc_auc_score(y, s), "prec": y[p].mean(),
           "lr": tpr / fpr if fpr > 0 else float("inf"), "flag": p.mean()}
    if pid is not None:
        # Prompt-clustered: resample PROMPTS. Rollouts within a prompt are
        # correlated, so a naive per-rollout bootstrap understates the width.
        rng = np.random.default_rng(0)
        up = np.unique(pid)
        idx = {q: np.where(pid == q)[0] for q in up}
        f = p.astype(float)
        draws = np.empty(N_BOOT_CHECK)
        for b in range(N_BOOT_CHECK):
            take = rng.choice(up, len(up), replace=True)
            draws[b] = f[np.concatenate([idx[q] for q in take])].mean()
        out["flag_lo"], out["flag_hi"] = np.percentile(draws, [2.5, 97.5])
    return out


def probe_series():
    P = pickle.load(open(PROBE, "rb"))
    out = {}
    thr = None
    for st in STEPS:
        X = np.load(f"{ACTS}/{st}/16.npy")
        y = pd.read_parquet(f"{ACTS}/{st}/labels.parquet")["last_block"].values.astype(int)
        s = P.predict_proba(X)[:, 1]
        pid = pd.read_parquet(f"{ACTS}/{st}/labels.parquet")["prompt_idx"].values
        if thr is None:
            thr = float(np.quantile(s, 0.5))
        out[st] = stats(y, s, thr, pid)
    return out


def judge_series():
    rows, thr = {}, None
    for st in STEPS:
        d = [json.loads(l) for l in open(f"{JUDGE}/step_{st}.jsonl")]
        y = np.array([r["correct"] for r in d])
        s = np.array([r["judge_score"] for r in d])
        pid = np.array([r["prompt_idx"] for r in d])
        if thr is None:
            thr = float(np.quantile(s, 0.5))
        rows[st] = stats(y, s, thr, pid)
    return rows


def parse(tex, which):
    """Pull the table whose HEADER names `which` AUROC. Header, not position --
    position is exactly what the swap bug got wrong."""
    hdr = re.search(r"RLOO step & \\textbf\{" + which + r" AUROC\}.*?\\bottomrule",
                    tex, re.S)
    if not hdr:
        sys.exit(f"FAIL: no table with a '{which} AUROC' header in the tex")
    body = hdr.group(0)
    cap = tex[hdr.end():hdr.end() + 700]
    caption = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\n\\end\{table\})", cap, re.S)
    rows = {}
    for line in body.split("\n"):
        m = re.match(r"\s*(?:\\textbf\{)?(\d+)\}?\s*&(.*)\\\\", line)
        if not m:
            continue
        cells = [re.sub(r"\\textbf\{|\}|\s", "", c) for c in m.group(2).split("&")]
        auroc = float(cells[0].split("[")[0])
        flag, fci = cells[4].split("[")
        flo, fhi = (float(x) for x in fci.rstrip("]").split(","))
        rows[int(m.group(1))] = {"auroc": auroc, "prec": float(cells[2]),
                                 "lr": float(cells[3]), "flag": float(flag),
                                 "flag_lo": flo, "flag_hi": fhi}
    return rows, (caption.group(1)[:80] if caption else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=os.path.join(ROOT, "writeup_judge.tex"))
    a = ap.parse_args()
    tex = open(a.tex).read()

    bad = 0
    for which, truth in (("probe", probe_series), ("judge", judge_series)):
        pub, caption = parse(tex, which)
        real = truth()
        print(f"\n=== {which} table ({len(pub)} rows) ===")
        print(f"  caption: {caption[:70]}...")
        for st in sorted(pub):
            for k in ("auroc", "prec", "lr", "flag", "flag_lo", "flag_hi"):
                p, r = pub[st][k], real[st][k]
                if abs(p - r) > TOL[k]:
                    print(f"  MISMATCH step {st:>3} {k:>5}: paper {p} vs recomputed {r:.4f}")
                    bad += 1
        if not bad:
            print(f"  all {6*len(pub)} cells match recomputation")

    # the swap check: the two tables must not be the same numbers
    pp, _ = parse(tex, "probe"); jj, _ = parse(tex, "judge")
    if all(abs(pp[s]["auroc"] - jj[s]["auroc"]) < 1e-9 for s in pp):
        print("\n  FAIL: the two tables are identical -- a table was overwritten")
        bad += 1

    print("\nAll table cells verified." if not bad else f"\n{bad} DISAGREEMENTS.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
