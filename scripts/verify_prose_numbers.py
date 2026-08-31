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
    from sklearn.metrics import roc_auc_score as _auc
    _pos = {st: [r["judge_score"] for r in D[st] if r["correct"] == 1] for st in (0, 99)}
    _neg = {st: [r["judge_score"] for r in D[st] if r["correct"] == 0] for st in (0, 99)}
    def auc_swap(pst, nst):
        """AUROC with positives from one checkpoint and negatives from another.
        This is what shows the flat end-to-end AUROC is two effects cancelling."""
        return _auc([1]*len(_pos[pst]) + [0]*len(_neg[nst]), _pos[pst] + _neg[nst])
    strat = {}
    for b in ("CORRECT", "wrong numbers used"):
        for st in (0, 99):
            sel = [r for r in D[st] if bucket(r) == b]
            strat[(b, st)] = np.mean([r["judge_score"] >= thr for r in sel])

    # --- quantities added by the corrected-mechanism section -------------
    _prompts = np.array(sorted({r["prompt_idx"] for r in D[0]}))
    _byp = {st: {q: [r for r in D[st] if r["prompt_idx"] == q] for q in _prompts}
            for st in (0, 99)}
    _rng = np.random.default_rng(0)
    _draws = [_rng.choice(_prompts, len(_prompts), True) for _ in range(2000)]

    def _auc_draw(st, draw):
        rows = [r for q in draw for r in _byp[st][q]]
        return _auc([r["correct"] for r in rows], [r["judge_score"] for r in rows])
    _dboot = np.array([_auc_draw(99, d) - _auc_draw(0, d) for d in _draws])
    auroc_delta = aur[99] - aur[0]
    auroc_lo, auroc_hi = np.percentile(_dboot, [2.5, 97.5])

    def _strat_p(b):
        def rate(st, draw):
            rows = [r for q in draw for r in _byp[st][q] if bucket(r) == b]
            return np.mean([r["judge_score"] >= thr for r in rows]) if rows else np.nan
        d = np.array([rate(99, dr) - rate(0, dr) for dr in _draws])
        d = d[~np.isnan(d)]
        return max(2 * min((d < 0).mean(), (d > 0).mean()), 1 / len(d))
    p_corr = _strat_p("CORRECT")
    p_ill = _strat_p("wrong numbers used")
    p_mid = _strat_p("valid form, wrong value")

    # flag rate at its minimum, and the two relative falls
    _f = {st: np.mean([r["judge_score"] >= thr
                       for r in [json.loads(l) for l in open(f"{JS}/step_{st}.jsonl")]])
          for st in (80,)}
    flag_min = _f[80]
    rel_acc = 100 * (y0.mean() - y9.mean()) / y0.mean()
    rel_flag = 100 * (p0.mean() - p9.mean()) / p0.mean()
    # composition of the WRONG-answer pool
    _ill = lambda st: (len([r for r in D[st] if bucket(r) == "wrong numbers used"])
                       / len([r for r in D[st] if r["correct"] == 0]))
    ill0, ill9 = 100 * _ill(0), 100 * _ill(99)
    prec_drop = 100 * (y0[p0].mean() - y9[p9].mean()) / y0[p0].mean()

    # --- format shift: the mechanism the corrected section rests on ----------
    import re as _re
    _canon = lambda e: _re.sub(r"[()\\s]", "", e or "")
    _key = {}
    for _st in (0, 99):
        for _r in D[_st]:
            if _r["equation"]:
                _key.setdefault((_r["prompt_idx"], _r["equation"]), []).append(_r["judge_score"])
    _rep = [v for v in _key.values() if len(v) > 1]
    n_repeat = len(_rep)
    n_straddle = sum(1 for v in _rep if (max(v) >= thr) != (min(v) >= thr))
    def _noparen(st):
        c = [r for r in D[st] if r["correct"] == 1]
        return sum(1 for r in c if "(" not in (r["equation"] or "")) / len(c)
    noparen0, noparen99 = _noparen(0), _noparen(99)
    _g = {}
    for _st in (0, 99):
        for _r in D[_st]:
            if _r["correct"] == 1 and _r["equation"]:
                k = (_r["prompt_idx"], _canon(_r["equation"]))
                _g.setdefault(k, {"p": [], "n": []})
                _g[k]["p" if "(" in _r["equation"] else "n"].append(_r["judge_score"] >= thr)
    _pairs = [(np.mean(g["p"]), np.mean(g["n"])) for g in _g.values() if g["p"] and g["n"]]
    paren_yes = np.mean([a for a, _ in _pairs])
    paren_no = np.mean([b for _, b in _pairs])

    # Some claims live only in the longer cut. Each entry names the papers it
    # applies to, so dropping a sentence from one paper does not fail the other.
    ALL = ("spotlight", "shortpaper", "interpscience")
    LONG = ("shortpaper",)
    SPOT = ("spotlight",)   # the corrected fixed-judge mechanism, 2pp only so far
    # The interpscience cut carries the corrected judge mechanism and the
    # LR+/lift material, so those checks apply to it too. Scoped one by one
    # rather than promoted to ALL: the shorter cuts genuinely drop some.
    LONG_I = ("shortpaper", "interpscience")
    SPOT_I = ("spotlight", "interpscience")
    CHECKS = [
        # (label, recomputed, as written in the paper, tolerance, which papers)
        ("accuracy loss, pp",        100*(y0.mean()-y9.mean()), "31.7",  0.15,  ALL),
        ("AUROC at step 0",          aur[0],                    "0.785", 0.0015, ALL),
        ("AUROC at step 99",         aur[99],                   "0.769", 0.0015, ALL),
        ("largest AUROC dip",        aur[0]-min(aur.values()),  "0.031", 0.002, LONG_I),
        # the spotlight writes these as percentages of the pool, the 3pp version
        # as accuracies; check whichever form the paper actually uses
        ("share correct, step 0",    100*y0.mean(),             "54.4",  0.15,  ALL),
        ("share correct, step 99",   100*y9.mean(),             "22.6",  0.15,  ALL),
        ("accuracy at step 40",      y4.mean(),                 "0.457", 0.0015, LONG_I),
        ("precision at step 0",      y0[p0].mean(),             "0.746", 0.0015, ALL),
        ("precision at step 99",     y9[p9].mean(),             "0.429", 0.0015, ALL),
        ("LR+ at step 0",            lr(y0, p0),                "2.46",  0.01, LONG_I),
        ("LR+ at step 99",           lr(y9, p9),                "2.57",  0.01, LONG_I),
        ("flag rate at step 0",      p0.mean(),                 "0.500", 0.0015, ALL),
        ("flag rate at step 99",     p9.mean(),                 "0.317", 0.0015, ALL),
        ("flag from prevalence",     fpred,                     "0.371", 0.0015, SPOT),
        ("prevalence share of move", 100*(fpred-p0.mean())/(p9.mean()-p0.mean()), "70", 1.0, LONG_I),
        ("lift confound share",      100*((tpr/fpred)-l0)/(l9-l0), "91",  1.0, LONG_I),
        ("pass off by 1--3",         pr(band(1, 3)),            "52.7",  0.15,  LONG),
        ("pass off by 50+",          pr(band(50, 1e18)),        "19.9",  0.15,  LONG),
        ("total rollouts",           sum(len(D[s]) for s in STEPS), "35{,}728", 0, ALL),
        ("problems",                 len({r["prompt_idx"] for r in D[0]}), "406", 0, ALL),
        # the corrected mechanism: the flat AUROC is a cancellation
        ("AUROC, step-99 negatives", auc_swap(0, 99),           "0.809", 0.002, LONG),
        ("AUROC, step-99 positives", auc_swap(99, 0),           "0.737", 0.002, LONG),
        ("pass on correct, step 0",  strat[("CORRECT", 0)],     "0.686", 0.0015, SPOT),
        ("pass on correct, step 99", strat[("CORRECT", 99)],    "0.600", 0.0015, SPOT),
        ("pass on illegal, step 0",  strat[("wrong numbers used", 0)],  "0.057", 0.0015, SPOT),
        ("pass on illegal, step 99", strat[("wrong numbers used", 99)], "0.013", 0.0015, SPOT),
        # the corrected mechanism and the alarm's own weaknesses
        ("AUROC change, end to end", auroc_delta,              "-0.017", 0.002, ALL),
        ("AUROC CI low",            auroc_lo,                  "-0.048", 0.004, ALL),
        ("AUROC CI high",           auroc_hi,                  "+0.014", 0.004, ALL),
        ("p, correct stratum",      p_corr,                    "0.013", 0.006, ALL),
        ("p, illegal stratum",      p_ill,                     "0.004", 0.0005, LONG),
        ("p, middle stratum",       p_mid,                     "0.74",  0.10,  ALL),
        ("flag rate minimum",       flag_min,                  "0.294", 0.0015, ALL),
        ("relative accuracy fall",  rel_acc,                   "58",    1.0,   ALL),
        ("relative flag fall",      rel_flag,                  "37",    1.0,   ALL),
        ("illegal share of wrong, 0",  ill0,                   "28",    1.0,   ALL),
        ("illegal share of wrong, 99", ill9,                   "43",    1.0,   ALL),
        ("precision drop, percent", prec_drop,                 "42",    1.0,   ALL),
        # the corrected mechanism: a fixed judge, changed inputs
        ("repeated (prompt,eqn) keys", n_repeat,               "965",   0, SPOT_I),
        ("of those, straddling thr",   n_straddle,             "one",   None, SPOT_I),
        ("no-paren share, step 0",     noparen0,               "0.017", 0.0015, SPOT_I),
        ("no-paren share, step 99",    noparen99,              "0.982", 0.0015, SPOT_I),
        ("matched: with parentheses",  paren_yes,              "0.746", 0.0015, SPOT_I),
        ("matched: without",           paren_no,               "0.548", 0.0015, SPOT_I),
    ]

    which = ("spotlight" if "spotlight" in a.tex
             else "interpscience" if "interpscience" in a.tex else "shortpaper")
    bad = 0
    print(f"  {'quantity':<26}{'recomputed':>12}{'in paper':>11}   status")
    for label, got, written, tol, applies in CHECKS:
        if which not in applies:
            continue
        if tol is None:            # spelled-out count, checked for presence only
            print(f"  {label:<26}{got:>12.0f}{written:>11}   "
                  f"{'OK' if (got == 1 and written in prose) else 'CHECK'}")
            bad += 0 if (got == 1 and written in prose) else 1
            continue
        want = float(written.replace("{,}", "").replace(",", "").lstrip("+"))
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
