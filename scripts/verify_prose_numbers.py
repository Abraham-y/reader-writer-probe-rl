"""Check the numbers written in the paper's PROSE, not just its tables.

WHY THIS EXISTS
verify_paper_tables.py checks table cells. It does not read a sentence. When
the 2pp spotlight was rewritten from scratch, roughly twenty numbers were
retyped into prose where nothing checked them, which is the exact failure this
project keeps hitting: the artifacts stay right and the prose drifts off them.

Each entry below is checked TWICE. The value must match a recomputation from
the rollout data, and it must literally appear in the paper. So a stale
sentence fails even when the underlying analysis is fine, and a changed
dataset fails even when the sentence was never touched.

    python scripts/verify_prose_numbers.py --tex writeup_judge_spotlight.tex
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from verify_judge_errors import bucket, SCORES as JS  # noqa: E402
from evaluation.countdown import evaluate_equation     # noqa: E402

STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_judge_spotlight.tex")
    a = ap.parse_args()
    tex = open(os.path.join(_ROOT, a.tex)).read()
    prose = tex[tex.index(r"\begin{abstract}"):tex.index(r"\begin{thebibliography}")]
    prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", prose, flags=re.S)

    D = {st: [json.loads(l) for l in open(f"{JS}/step_{st}.jsonl")] for st in STEPS}
    thr = float(np.quantile([r["judge_score"] for r in D[0]], 0.5))
    ar = lambda st: (np.array([r["correct"] for r in D[st]]),
                     np.array([r["judge_score"] for r in D[st]]))
    y0, s0 = ar(0); y9, s9 = ar(99); y4, _ = ar(40)
    p0, p9 = s0 >= thr, s9 >= thr
    tpr, fpr = p0[y0 == 1].mean(), p0[y0 == 0].mean()
    b9 = y9.mean()
    fpred = b9 * tpr + (1 - b9) * fpr
    lr = lambda y, p: p[y == 1].mean() / p[y == 0].mean()
    aur = {st: roc_auc_score(*ar(st)) for st in STEPS}
    wrong = [r for r in D[0] if bucket(r) == "valid form, wrong value"]
    band = lambda lo, hi: [r for r in wrong
                           if lo <= abs(evaluate_equation(r["equation"]) - int(r["target"])) < hi]
    pr = lambda rows: 100 * np.mean([r["judge_score"] >= thr for r in rows])
    l0 = y0[p0].mean() / y0.mean(); l9 = y9[p9].mean() / y9.mean()

    # (label, recomputed, as-written-in-the-paper, tolerance)
    CHECKS = [
        ("accuracy loss, pp",        100*(y0.mean()-y9.mean()), "31.7",  0.15),
        ("AUROC at step 0",          aur[0],                    "0.785", 0.0015),
        ("AUROC at step 99",         aur[99],                   "0.769", 0.0015),
        ("largest AUROC dip",        aur[0]-min(aur.values()),  "0.031", 0.002),
        ("accuracy at step 0",       y0.mean(),                 "0.544", 0.0015),
        ("accuracy at step 40",      y4.mean(),                 "0.457", 0.0015),
        ("precision at step 0",      y0[p0].mean(),             "0.746", 0.0015),
        ("precision at step 99",     y9[p9].mean(),             "0.429", 0.0015),
        ("LR+ at step 0",            lr(y0, p0),                "2.46",  0.01),
        ("LR+ at step 99",           lr(y9, p9),                "2.57",  0.01),
        ("flag rate at step 0",      p0.mean(),                 "0.500", 0.0015),
        ("flag rate at step 99",     p9.mean(),                 "0.317", 0.0015),
        ("flag from prevalence",     fpred,                     "0.371", 0.0015),
        ("prevalence share of move", 100*(fpred-p0.mean())/(p9.mean()-p0.mean()), "70", 1.0),
        ("lift confound share",      100*((tpr/fpred)-l0)/(l9-l0), "91",  1.0),
        ("pass off by 1--3",         pr(band(1, 3)),            "52.7",  0.15),
        ("pass off by 50+",          pr(band(50, 1e18)),        "19.9",  0.15),
        ("total rollouts",           sum(len(D[s]) for s in STEPS), "35{,}728", 0),
        ("problems",                 len({r["prompt_idx"] for r in D[0]}), "406", 0),
    ]

    bad = 0
    print(f"  {'quantity':<26}{'recomputed':>12}{'in paper':>11}   status")
    for label, got, written, tol in CHECKS:
        want = float(written.replace("{,}", "").replace(",", ""))
        num_ok = abs(got - want) <= tol
        # is it actually written down? skip the presence test for values that
        # legitimately appear only inside the table (none here, but be explicit)
        in_paper = re.search(re.escape(written), prose) is not None
        status = "OK" if (num_ok and in_paper) else (
            "VALUE DRIFTED" if not num_ok else "NOT IN PROSE")
        if status != "OK":
            bad += 1
        print(f"  {label:<26}{got:>12.3f}{written:>11}   {status}")
    print()
    print("  every number written in the prose matches the data." if not bad
          else f"  {bad} PROBLEMS -- do not submit until this is green.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
