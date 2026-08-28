"""Why the judge is wrong, and why its AUROC cannot see the collapse.

WHAT THIS COMPUTES
Every rollout the judge scored is sorted into four mutually exclusive buckets
by the exact Countdown verifier:

    CORRECT                  a valid equation that hits the target
    valid form, wrong value  uses the given numbers legally, evaluates wrong
    wrong numbers used       uses numbers it was not given, or reuses one
    no answer given          no parseable <answer> block

Then, per checkpoint, two things: what SHARE of rollouts falls in each bucket
(the population), and how often the judge PASSES each bucket at the frozen
step-0 threshold (its conditional behaviour).

WHY IT MATTERS
The paper claims the judge's discrimination survives a 31.7pp accuracy
collapse because the failure is a prevalence shift, not an evaluator failure.
This is the direct evidence: the conditional pass rates barely move while the
population moves a great deal. AUROC is a function of the conditional score
distributions alone, so it is blind to the second by construction.

It also answers the first question any reader asks -- why is the judge wrong
at all? It checks rules, not arithmetic. It rejects illegal-number answers
almost always, and coin-flips on well-formed answers that simply evaluate to
the wrong number, with only a coarse sense of magnitude.

    python scripts/verify_judge_errors.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402

SCORES = os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag")
STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
BUCKETS = ["CORRECT", "valid form, wrong value", "wrong numbers used", "no answer given"]

# Published in the paper. Any drift here fails the gate.
PUBLISHED = {
    #        share of rollouts                    judge pass rate
    #        corr   wrong-val  wrong-num          corr   wrong-val  wrong-num
    0:  {"share": (0.544, 0.294, 0.129), "pass": (0.686, 0.408, 0.057)},
    99: {"share": (0.226, 0.442, 0.330), "pass": (0.600, 0.399, 0.013)},
}
TOL = 0.002


def bucket(r) -> str:
    eq = r["equation"]
    if eq is None:
        return "no answer given"
    if not validate_equation(eq, list(r["nums"])):
        return "wrong numbers used"
    v = evaluate_equation(eq)
    if v is None:
        return "wrong numbers used"
    return "CORRECT" if abs(v - int(r["target"])) < 1e-5 else "valid form, wrong value"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows0 = [json.loads(l) for l in open(f"{SCORES}/step_0.jsonl")]
    thr = float(np.quantile([r["judge_score"] for r in rows0], 0.5))

    L = [f"Judge error decomposition. Frozen threshold = step-0 median = {thr:.4f}.", ""]
    L.append(f"  {'step':>4}" + "".join(f"{b[:13]:>15}" for b in BUCKETS[:3]))
    L.append(f"  {'':>4}" + "".join(f"{'share   pass':>15}" for _ in BUCKETS[:3]))
    per = {}
    for st in STEPS:
        rows = [json.loads(l) for l in open(f"{SCORES}/step_{st}.jsonl")]
        n = len(rows)
        share, passed = {}, {}
        for b in BUCKETS:
            sel = [r for r in rows if bucket(r) == b]
            share[b] = len(sel) / n
            passed[b] = (sum(r["judge_score"] >= thr for r in sel) / len(sel)) if sel else float("nan")
        per[st] = {"share": share, "pass": passed, "n": n}
        L.append(f"  {st:>4}" + "".join(
            f"{share[b]:>8.1%}{passed[b]:>7.1%}" for b in BUCKETS[:3]))

    L += ["", "  Read down a 'pass' column: the judge's conditional behaviour.",
          "  Read down a 'share' column: the population it is handed.",
          "  The first barely moves; the second moves a great deal. AUROC is a",
          "  function of the first alone, which is why it reports nothing.", ""]

    # how wrong is 'wrong'? does the judge track magnitude?
    bands = [(0, 1), (1, 3), (3, 10), (10, 50), (50, float("inf"))]
    hits = {b: [0, 0] for b in bands}
    for r in rows0:
        if bucket(r) != "valid form, wrong value":
            continue
        d = abs(evaluate_equation(r["equation"]) - int(r["target"]))
        for b in bands:
            if b[0] <= d < b[1]:
                hits[b][0] += 1
                hits[b][1] += r["judge_score"] >= thr
    L.append("  Step 0, well-formed but wrong. Does the judge notice HOW wrong?")
    for (lo, hi), (n, p) in hits.items():
        if n:
            lab = f"off by {lo}--{hi}" if hi != float("inf") else "off by 50+"
            L.append(f"    {lab:<16}{n:>5}{p/n:>9.1%}")
    L += ["", "  It rejects wild misses and coin-flips everything near the target:",
          "  a coarse magnitude check, not arithmetic.", ""]

    bad = 0
    for st, exp in PUBLISHED.items():
        got_s = tuple(per[st]["share"][b] for b in BUCKETS[:3])
        got_p = tuple(per[st]["pass"][b] for b in BUCKETS[:3])
        for name, got, want in (("share", got_s, exp["share"]), ("pass", got_p, exp["pass"])):
            for g, w in zip(got, want):
                if abs(g - w) > TOL:
                    L.append(f"  MISMATCH step {st} {name}: published {w} vs recomputed {g:.4f}")
                    bad += 1
    L.append("  all published values match recomputation." if not bad else f"  {bad} DISAGREEMENTS.")

    out = "\n".join(L)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
