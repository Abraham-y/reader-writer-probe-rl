"""Check the numbers written in the paper's PROSE, not just its tables.

WHY THIS EXISTS
verify_tables.py checks table cells. It does not read a sentence. When
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
import paper_ledger as L                                 # noqa: E402

STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    a = ap.parse_args()
    tex = open(os.path.join(_ROOT, a.tex)).read()
    # Body AND appendix. Appendices sit after the bibliography, and when
    # material moves there it must not fall out of this check -- an ungated
    # region is how every number this suite exists to protect went wrong before.
    prose = L.canonical(tex)   # the text verify_numbers.py checks coverage on

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
    # strip parentheses and whitespace (an earlier r"[()\\s]" stripped a
    # backslash and the letter s instead of whitespace)
    _canon = lambda e: _re.sub(r"[()\s]", "", e or "")
    # The fixed-function check, over EVERY checkpoint the judge scored. An
    # earlier version keyed only steps 0 and 99 and the paper reported that
    # subset as if it were the whole.
    _key, _seen_at = {}, {}
    for _st in STEPS:
        for _r in D[_st]:
            if _r["equation"]:
                k = (_r["prompt_idx"], _r["equation"])
                _key.setdefault(k, []).append(_r["judge_score"])
                _seen_at.setdefault(k, set()).add(_st)
    _rep = [v for v in _key.values() if len(v) > 1]
    n_repeat = len(_rep)
    n_straddle = sum(1 for v in _rep if (max(v) >= thr) != (min(v) >= thr))
    max_straddle_gap = max([max(abs(x - thr) for x in v) for v in _rep
                            if (max(v) >= thr) != (min(v) >= thr)] or [0.0])
    _both = [k for k, st in _seen_at.items() if 0 in st and 99 in st]
    n_both = len(_both)
    n_both_straddle = sum(1 for k in _both
                          if (max(_key[k]) >= thr) != (min(_key[k]) >= thr))
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

    # The bracket effect, with a prompt-clustered bootstrap over the matched
    # pairs (seed 0, 2,000 draws). An earlier interval was typed in from notes.
    _pair_list = [(k[0], np.mean(g["p"]) - np.mean(g["n"])) for k, g in _g.items() if g["p"] and g["n"]]
    _pp = np.array([q for q, _ in _pair_list]); _dd = np.array([d for _, d in _pair_list])
    _uq = np.unique(_pp); _rng_b = np.random.default_rng(0); _bb = []
    for _ in range(2000):
        _w = np.bincount(np.searchsorted(_uq, _rng_b.choice(_uq, len(_uq))), minlength=len(_uq))
        _wt = _w[np.searchsorted(_uq, _pp)]
        _bb.append((_wt * _dd).sum() / _wt.sum())
    paren_lo, paren_hi = np.percentile(_bb, [2.5, 97.5])

    # (label, recomputed, as written, tolerance, context). Every check names the
    # sentence it belongs to: {n} is the number, the rest pins it in place, so
    # the same digits elsewhere cannot satisfy it (the 2026-10-10 mutation test
    # passed 23 planted errors through bare-number checks).
    CHECKS = [
        ("accuracy loss, pp",           100*(y0.mean()-y9.mean()), "31.7",  0.15,  r"a drop of {n} points"),
        ("abstract: judge-ladder accuracy, step 0 (%)", 100*y0.mean(), "54", 0.5, r"accuracy falls from {n}\\% to 23\\%"),
        ("abstract: judge-ladder accuracy, step 99 (%)", 100*y9.mean(), "23", 0.5, r"falls from 54\\% to {n}\\%"),
        ("relative accuracy fall",      rel_acc,                   "58",    0.5,   r"points, or {n}\\%"),
        ("judge AUROC at step 0",       aur[0],                    "0.785", 0.0015, r"It is {n} at step 0"),
        ("judge AUROC at step 0, Table 1", aur[0],                 "0.785", 0.0015, r"its {n} is on"),
        ("judge AUROC at step 99",      aur[99],                   "0.769", 0.0015, r"and {n} at step 99"),
        ("judge AUROC dip",             aur[30]-aur[0],            "-0.031", 0.0015, r"a dip of {n} at step 30"),
        ("judge AUROC at step 50",      aur[50],                   "0.778", 0.0015, r"at step 50 it is {n}"),
        ("accuracy at step 0",          y0.mean(),                 "0.544", 0.0015, r"true accuracy of {n}"),
        ("accuracy, steps 0 to 40",     y4.mean(),                 "0.457", 0.0015, r"falls 0\.544 \\to {n}"),
        ("accuracy at step 0 (dip)",    y0.mean(),                 "0.544", 0.0015, r"falls {n} \\to 0\.457"),
        ("accuracy, steps 0 to 99",     y9.mean(),                 "0.226", 0.0015, r"fell 0\.544 \\to {n}"),
        ("accuracy at step 0 (end)",    y0.mean(),                 "0.544", 0.0015, r"fell {n} \\to 0\.226"),
        ("correct share, step 0",       y0.mean(),                 "0.544", 0.0015, r"answers falls {n} \\to 0\.226"),
        ("correct share, step 99",      y9.mean(),                 "0.226", 0.0015, r"answers falls 0\.544 \\to {n}"),
        ("precision at step 0",         y0[p0].mean(),             "0.746", 0.0015, r"precision {n} at its own"),
        ("precision fall, from",        y0[p0].mean(),             "0.746", 0.0015, r"\({n} \\to 0\.429\)"),
        ("precision at step 99",        y9[p9].mean(),             "0.429", 0.0015, r"\(0\.746 \\to {n}\)"),
        ("precision drop, percent",     prec_drop,                 "42",    0.5,   r"by {n}\\% \(0\.746"),
        ("LR+ at step 0",               lr(y0, p0),                "2.46",  0.005, r"\({n} \\to 2\.57, ratio"),
        ("LR+ at step 99",              lr(y9, p9),                "2.57",  0.005, r"2\.46 \\to {n}, ratio"),
        ("TPR at step 0",               tpr,                       "0.686", 0.0015, r"TPR\} {n} \\to"),
        ("TPR at step 99",              p9[y9 == 1].mean(),        "0.600", 0.0015, r"TPR\} 0\.686 \\to {n}"),
        ("FPR at step 0",               fpr,                       "0.279", 0.0015, r"FPR\} {n} \\to 0\.234"),
        ("FPR at step 99",              p9[y9 == 0].mean(),        "0.234", 0.0015, r"FPR\} 0\.279 \\to {n}"),
        ("judge flag rate, 0 to 99",    p0.mean(),                 "0.500", 0.0015, r"decreases rather than increases, {n} \\to 0\.317"),
        ("judge flag rate at step 99",  p9.mean(),                 "0.317", 0.0015, r"increases, 0\.500 \\to {n}"),
        ("judge flag rate minimum",     flag_min,                  "0.294", 0.0015, r"minimum of {n} at step 80"),
        ("prevalence share of flag move", 100*(fpred-p0.mean())/(p9.mean()-p0.mean()), "70", 0.5, r"predicts about {n}\\% of it"),
        ("prevalence share (App B)",    100*(fpred-p0.mean())/(p9.mean()-p0.mean()), "70", 0.5, r"roughly {n}\\% of the flag"),
        ("lift confound share",         100*((tpr/fpred)-l0)/(l9-l0), "91", 0.5, r"{n}\\% of lift's"),
        ("judge answers in total",      sum(len(D[st]) for st in STEPS), "35{,}728", 0, r"{n} answers in total"),
        ("judge answers, App A",        sum(len(D[st]) for st in STEPS), "35{,}728", 0, r"Two of the {n} answers"),
        ("AUROC change, end to end",    auroc_delta,               "-0.017", 0.0015, r"\({n}, 95\\% CI \[-0\.048"),
        ("AUROC change CI low",         auroc_lo,                  "-0.048", 0.0015, r"95\\% CI \[{n}, \+0\.014\]"),
        ("AUROC change CI high",        auroc_hi,                  "+0.014", 0.0015, r"\[-0\.048, {n}\]\), though"),
        ("repeated (problem, eqn) pairs", n_repeat,                "4{,}651", 0, r"it scored {n} \(problem, equation\)"),
        ("repeated pairs, App A",       n_repeat,                  "4{,}651", 0, r"Of the {n} repeated pairs"),
        ("of those, straddling",        n_straddle,                "8",     0,     r"; {n} cross the threshold"),
        ("straddlers' distance from it", max_straddle_gap,         "3",     None,  r"all within {n}\\times10\^\{-4\} of it"),
        ("pairs at steps 0 and 99",     n_both,                    "48",    0,     r"of the {n} pairs scored at both"),
        ("repeats identical every time (%)", 100*np.mean([max(v) == min(v) for v in _rep]), "79", 0.5, r"{n}\\% score identically every time"),
        ("repeats differing by more than 0.01", sum(max(v) - min(v) > 0.01 for v in _rep), "109", 0, r"and {n} differ by more than 0\.01"),
        ("largest difference between repeats", max(max(v) - min(v) for v in _rep), "0.109", 0.0005, r"the largest by {n}\."),
        ("no-paren share, step 0",      noparen0,                  "0.017", 0.0015, r"none rising {n} \\to"),
        ("no-paren share, step 99",     noparen99,                 "0.982", 0.0015, r"rising 0\.017 \\to {n}"),
        ("bracketed forms passed",      paren_yes,                 "0.746", 0.0015, r"passes {n} of the bracketed"),
        ("bare forms passed",           paren_no,                  "0.548", 0.0015, r"and {n} of the bare"),
        ("bracket effect",              paren_yes - paren_no,      "+0.198", 0.0015, r"\({n}, 95\\% CI \[\+0\.1"),
        ("bracket effect CI low",       paren_lo,                  "+0.127", 0.0015, r"95\\% CI \[{n}, \+0\.2\d\d\]\)\. The judge"),
        ("bracket effect CI high",      paren_hi,                  "+0.266", 0.0015, r"\[\+0\.1\d\d, {n}\]\)\. The judge"),
    ]
    # the one count written as a word
    word_ok = n_both_straddle == 0 and re.search(r"48 pairs scored at both step 0 and step 99, none do", prose) is not None
    L.claim("pairs at steps 0 and 99 that cross: none", r"of the 48 pairs scored at both step 0 and step 99, none do", word_ok)
    # the straddlers all sit within 3e-4 of the threshold
    gap_ok = max_straddle_gap < 3e-4

    bad = 0 if word_ok else 1
    print(f"  {'quantity':<32}{'recomputed':>12}{'in paper':>11}   status")
    for label, got, written, tol, ctx in CHECKS:
        pat = ctx.replace("{n}", L.number_pattern(written))
        in_paper = re.search(pat, prose) is not None
        if tol is None:                       # the straddle gap, an upper bound
            num_ok = gap_ok
        else:
            want = float(written.replace("{,}", "").replace(",", "").lstrip("+"))
            w_ = written.replace("{,}", "").lstrip("+-")
            tol = min(tol, 0.5 * 10 ** -(len(w_.split(".")[1]) if "." in w_ else 0) + 1e-9)   # exact rounding
            num_ok = abs(got - want) <= tol
        L.claim(label, pat, num_ok and in_paper)
        status = "OK" if (num_ok and in_paper) else ("VALUE DRIFTED" if not num_ok else "NOT IN PAPER")
        bad += status != "OK"
        print(f"  {label:<32}{got:>12.4f}{written:>11}   {status}")
    L.dump("verify_prose_numbers")
    print()
    print("  every number written in the prose matches the data." if not bad
          else f"  {bad} PROBLEMS -- do not submit until this is green.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
