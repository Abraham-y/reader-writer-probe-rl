#!/usr/bin/env python3
"""Gate the numbers and the front matter that the InterpScience camera-ready added.

WHY THIS EXISTS
The camera-ready answers two reviews, and every answer is a new number in prose
or a new table: the accuracy fall that precedes the AUROC change, the agreement
between the two ladders, the lag recomputed on prompts the reward probe never
saw, and the protocol facts in Table 1(b). This project's recurring defect is a
number typed into prose that no script checks, so each of them is recomputed
here and must also appear literally in the paper.

It also replaces the anonymity scan for this paper, which a de-anonymised
camera-ready must now fail, with the inverse check: the real author block, the
final-mode style, the workshop's notice, and no leftover blind-review wording.
And it fails while any `CAMERA-READY TODO` remains in the tex, so an open item
cannot be uploaded by accident.

    python scripts/verify_camera_ready.py [--tex writeup_interpscience.tex]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "followup", "experiments", "fragility", "phase0_replicate"))

ACTS = os.path.join(_ROOT, "followup", "acts", "phase0_harvest_runA")
JUDGE = os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag")
CPJ = os.path.join(_ROOT, "followup", "results", "fragility", "changepoint_lag.json")
# check_everything.sh runs changepoint_lag.py first and writes here; read that
# fresh recomputation rather than the committed file whenever it exists.
CPJ_FRESH = os.environ.get("CHANGEPOINT_JSON", "/tmp/_cp.json")
CLEAN = os.path.join(_ROOT, "extension", "cache", "probe_cache_n500_clean406",
                     "C_outcome_l16_pre_answer.meta.json")
STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

bad: list[str] = []
skipped: list[str] = []


def check(label, got, written, tol, tex, ctx=None):
    """`got` must be within tol of `written`, and `written` must be in the paper.

    `ctx` is a regex around the number, written as {n}, for numbers whose digits
    also appear elsewhere in the paper for an unrelated reason (a table cell, a
    CI bound): without it the presence test could pass on the wrong occurrence.
    """
    num = float(written.replace("{,}", "").replace(",", "").replace("+", ""))
    ok_val = tol is None or abs(got - num) <= tol
    # exact token: the number must not sit inside a longer number
    pat = r"(?<![\d.])" + re.escape(written) + r"(?![\d])"
    if ctx is not None:
        pat = ctx.replace("{n}", pat)
    ok_tex = re.search(pat, tex) is not None
    flag = "OK" if ok_val and ok_tex else "MISMATCH"
    print(f"  {label:<48} recomputed {got:>9.4f}   paper {written:>14}   {flag}")
    if not ok_val:
        bad.append(f"{label}: recomputed {got:.4f}, paper says {written}")
    if not ok_tex:
        bad.append(f"{label}: '{written}' does not appear in the tex")


def paired_acc(per, s, prompts, n_boot=2000, seed=0):
    """Prompt-clustered paired bootstrap of accuracy(step s) - accuracy(step 0), in pp."""
    rng = np.random.default_rng(seed)
    a, b = per[0], per[s]
    pt = 100 * (b["sum"].sum() / b["count"].sum() - a["sum"].sum() / a["count"].sum())
    d = np.empty(n_boot)
    for i in range(n_boot):
        dr = rng.choice(len(prompts), len(prompts))
        aa, bb = a.iloc[dr], b.iloc[dr]
        d[i] = bb["sum"].sum() / bb["count"].sum() - aa["sum"].sum() / aa["count"].sum()
    lo, hi = 100 * np.percentile(d, [2.5, 97.5])
    return pt, lo, hi


def accuracy_lead(tex):
    print("\n=== accuracy falls before the AUROC moves (section 4.1) ===")
    lab = {s: pd.read_parquet(f"{ACTS}/{s}/labels.parquet") for s in STEPS}
    prompts = np.unique(lab[0].prompt_idx)
    check("first-block accuracy, step 0", lab[0].first_block.mean(), "0.546", 0.0015, tex)
    check("last-block accuracy, step 0", lab[0].last_block.mean(), "0.372", 0.0015, tex)
    for rule, s, w_pt, w_ci in (("first_block", 10, "5.2", "[-6.9, -3.7]"),
                                ("last_block", 10, "6.4", "[-8.2, -4.6]"),
                                ("first_block", 40, "10.0", None),
                                ("last_block", 40, "7.0", None)):
        per = {t: lab[t].groupby("prompt_idx")[rule].agg(["sum", "count"]).reindex(prompts)
               for t in (0, s)}
        pt, lo, hi = paired_acc(per, s, prompts)
        check(f"{rule} fall by step {s}, pp", -pt, w_pt, 0.05, tex)
        if w_ci:
            lo_w, hi_w = (float(x) for x in w_ci.strip("[]").split(","))
            ok = abs(lo - lo_w) <= 0.15 and abs(hi - hi_w) <= 0.15 and w_ci in tex
            print(f"  {rule + ' step-10 CI':<48} recomputed [{lo:+.1f}, {hi:+.1f}]   "
                  f"paper {w_ci:>14}   {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append(f"{rule} step-10 CI: recomputed [{lo:.2f}, {hi:.2f}], paper {w_ci}")
            if hi >= 0:
                bad.append(f"{rule} step-10 fall is no longer significant: CI [{lo:.2f}, {hi:.2f}]")

    # the two ladders are independent draws; their first-block accuracies must agree
    gap = 0.0
    for s in STEPS:
        j = [json.loads(l) for l in open(f"{JUDGE}/step_{s}.jsonl")]
        gap = max(gap, abs(lab[s].first_block.mean() - np.mean([r["correct"] for r in j])))
    check("max first-block gap, probe vs judge ladder", gap, "0.016", 0.0015, tex)

    cp = json.load(open(CPJ_FRESH if os.path.exists(CPJ_FRESH) else CPJ))["paired_auroc_diffs"]
    first = json.load(open(os.path.join(_ROOT, "followup", "results", "fragility", "changepoint_lag_firstblock.json")))
    check("first-block AUROC, step 0", first["per_step"]["0"]["auroc"], "0.919", 0.0015, tex)
    check("first-block AUROC change at step 20", -first["paired_auroc_diffs"]["0-20"]["delta"], "+0.017", 0.0015, tex)
    check("first-block AUROC change at step 40", -first["paired_auroc_diffs"]["0-40"]["delta"], "-0.043", 0.0015, tex)
    if first["paired_auroc_diffs"]["0-40"]["p"] >= 0.05 or first["paired_auroc_diffs"]["0-30"]["p"] < 0.05:
        bad.append("first-block AUROC no longer first moves at step 40")
    check("last-block AUROC, step 0", json.load(open(CPJ))["per_step"]["0"]["auroc"], "0.783", 0.0015, tex)
    pmin = min(cp[f"0-{s}"]["p"] for s in (10, 20, 30, 40))
    # p-values come from the suite's 500-draw recomputation; at p ~ 0.2 the
    # Monte-Carlo error of a 500-draw bootstrap p is ~0.02, so allow 0.04
    check("smallest AUROC p, steps 10-40", pmin, "0.22", 0.04, tex)
    check("AUROC change at step 50", -cp["0-50"]["delta"], "-0.086", 0.0015, tex)
    if pmin < 0.05 or cp["0-50"]["p"] >= 0.05:
        bad.append("the AUROC's flat window through step 40 / break at 50 no longer holds")


def disjoint(tex):
    print("\n=== the reward probe's fitting prompts vs the 406 evaluation prompts ===")
    # By content, not index: the two files number their prompts independently,
    # and comparing indices is what produced the retracted 248/158 split.
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    from verify_reward_ladder import fitting_overlap  # noqa: E402
    n = fitting_overlap()
    print(f"  shared prompts (sorted numbers + target): {n}")
    if n:
        bad.append(f"{n} of the reward probe's fitting prompts are among the 406")
    for stale in ("app:overlap", "never saw", "tab:overlap"):
        if stale in tex:
            bad.append(f"the retracted seen/unseen analysis is still referenced ({stale!r})")


def protocols(tex):
    print("\n=== the evaluation protocols of Table 1(b) and section 4.3 ===")
    files = {"headline": ["eval_c_outcome_n500.json", "eval_runA_postRL_n500.json",
                          "eval_runB_postRL_n500.json"],
             "arm": ["eval_armA_residual_step100.json", "eval_armB_surface_step100.json",
                     "eval_runB_armprotocol_step100.json", "eval_armRaw_step100.json"]}
    if not all(os.path.exists(os.path.join(_ROOT, f)) for fs in files.values() for f in fs):
        skipped.append("protocol checks (eval_*.json not present)")
        print("  SKIP: rollout files not present")
        return
    from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402
    clean = {r["prompt_idx"] for r in json.load(open(CLEAN))}
    ans = re.compile(r"<answer>(.*?)</answer>", re.S)

    def first_block(resp, nums, tgt):
        mm = ans.search(resp)
        if not mm:
            return 0
        eq = mm.group(1).strip()
        if not validate_equation(eq, list(nums)):
            return 0
        v = evaluate_equation(eq)
        return int(v is not None and abs(v - int(tgt)) < 1e-5)

    acc = {}
    for proto, fs in files.items():
        want = {"headline": 16, "arm": 8}[proto]
        for f in fs:
            rows = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
            k = {len(r["response"]) for r in rows}
            if k != {want}:
                bad.append(f"{f}: {k} answers per prompt, Table 1(b) says {want} ({proto})")
            ys = [first_block(x, r["nums"], r["target"])
                  for i, r in enumerate(rows) if i in clean for x in r["response"]]
            acc[f] = float(np.mean(ys))
    for proto, want in (("headline", 16), ("arm", 8)):
        ok = re.search(rf"^{proto} & {want} &", tex, re.M) is not None
        print(f"  {'Table 1(b): ' + proto + ' answers per prompt':<48} files {want}   "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"Table 1(b) row '{proto}' does not say {want} answers per prompt")

    check("C_outcome first-block, headline", acc["eval_c_outcome_n500.json"], "0.550", 0.0015, tex)
    check("runA first-block, headline", acc["eval_runA_postRL_n500.json"], "0.236", 0.0015, tex)
    check("runB first-block, headline (arm C's first score)", acc["eval_runB_postRL_n500.json"], "0.0734", 0.00015, tex)
    check("arm C (runB) first-block, arm protocol", acc["eval_runB_armprotocol_step100.json"], "0.0782", 0.00015, tex)
    check("arm R first-block, arm protocol", acc["eval_armRaw_step100.json"], "0.0514", 0.00015, tex)
    # Arm A's evaluation command was never recorded. Re-running the arm protocol
    # on its final checkpoint (and on arm B's, whose command was recorded) must
    # reproduce each evaluation file answer for answer; vLLM sampling is
    # deterministic for a fixed model and settings, so this pins the settings.
    for orig in ("eval_armA_residual_step100.json", "eval_armB_surface_step100.json"):
        re_ = orig.replace(".json", "_rescore.json")
        a_ = [json.loads(l) for l in open(os.path.join(_ROOT, orig)) if l.strip()]
        b_ = [json.loads(l) for l in open(os.path.join(_ROOT, re_)) if l.strip()]
        same = sum(x == y for p_, q_ in zip(a_, b_) for x, y in zip(p_["response"], q_["response"]))
        total = sum(len(p_["response"]) for p_ in a_)
        ok = same == total and len(a_) == len(b_) and "answer for answer" in tex
        print(f"  {'re-scored ' + orig.split('_')[1] + ' identical to original':<48} {same}/{total}   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{orig}: re-score reproduces {same}/{total} answers")
    check("arm A first-block, arm protocol", acc["eval_armA_residual_step100.json"], "0.1678", 0.00015, tex)
    check("arm B first-block, arm protocol", acc["eval_armB_surface_step100.json"], "0.0000", 0.00015, tex)
    # Table 2 puts all three arms on the arm protocol. Recompute its two contrasts
    # and the protocol's own effect on arm C with the arms gate's estimator.
    sys.path.insert(0, os.path.join(_ROOT, "extension", "probe"))
    import verify_residual_arms as arms_gate  # noqa: E402
    keep = arms_gate.clean_prompts()
    per = {f: arms_gate.load_arm(f, keep) for f in
           ("eval_armA_residual_step100.json", "eval_armB_surface_step100.json",
            "eval_runB_armprotocol_step100.json", "eval_runB_postRL_n500.json",
            "eval_armRaw_step100.json")}
    c_arm = per["eval_runB_armprotocol_step100.json"]
    a_arm = per["eval_armA_residual_step100.json"]
    for label, treat, ref, w_pt, w_ci in (
            ("arm B - arm A (Table 2)", "eval_armB_surface_step100.json", a_arm, "-16.78", "[-18.97,-14.66]"),
            ("arm C - arm A (Table 2)", "eval_runB_armprotocol_step100.json", a_arm, "-8.96", "[-10.81,-7.17]"),
            ("arm R - arm A (Table 2)", "eval_armRaw_step100.json", a_arm, "-11.64", "[-13.76,-9.61]"),
            ("arm A - arm R (section 4.3)", "eval_armA_residual_step100.json", per["eval_armRaw_step100.json"], "+11.64", "[+9.61, +13.76]"),
            ("arm B - arm R (section 4.3)", "eval_armB_surface_step100.json", per["eval_armRaw_step100.json"], "-5.14", "[-6.25, -4.09]"),
            ("arm C: arm - headline protocol", "eval_runB_armprotocol_step100.json",
             per["eval_runB_postRL_n500.json"], "+0.48", "[-0.62, +1.60]")):
        d, lo, hi = arms_gate.paired_bootstrap(per[treat], ref, 10000, 0)
        check(label + ", pp", d, w_pt, 0.006, tex)
        lo_w, hi_w = (float(x) for x in w_ci.strip("[]").split(","))
        ok = abs(lo - lo_w) < 0.1 and abs(hi - hi_w) < 0.1 and (w_ci in tex or "prose" in label)
        print(f"  {label + ' CI':<48} recomputed [{lo:+.2f}, {hi:+.2f}]   paper {w_ci:>14}   "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{label} CI: recomputed [{lo:.2f}, {hi:.2f}], paper {w_ci}")


def revision_numbers(tex):
    """Every number the 2026-10-01 audit revision introduced, recomputed."""
    print("\n=== numbers introduced by the audit revision ===")
    import pickle, re as _re
    sys.path.insert(0, os.path.join(_ROOT, "followup", "experiments", "fragility", "phase0_replicate"))
    from changepoint_lag import load_checkpoint  # noqa: E402
    probe = pickle.load(open(os.path.join(_ROOT, "extension", "cache", "steering",
                                          "probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl"), "rb"))
    lad = {s: load_checkpoint("phase0_harvest_runA", s, 16)[0] for s in (0, 10, 20, 30, 40)}
    check("mean probe score, step 0", lad[0].mean(), "0.473", 0.0015, tex)
    check("mean probe score, step 40", lad[40].mean(), "0.677", 0.0015, tex)
    check("share above 0.95, step 0 (%)", 100 * (lad[0] > 0.95).mean(), "5.2", 0.05, tex)
    check("share above 0.95, step 40 (%)", 100 * (lad[40] > 0.95).mean(), "38.8", 0.05, tex)
    t0 = float(np.quantile(lad[0], 0.5))
    check("attacked flag-rate change, step 30 (%)",
          100 * ((lad[30] >= t0).mean() / (lad[0] >= t0).mean() - 1), "+20.2", 0.05, tex)

    # the old control ladder, scored with the reward probe
    C = os.path.join(_ROOT, "followup", "acts", "vanilla_rloo_ladder")
    st = sorted(int(x) for x in os.listdir(C) if x.isdigit())
    sc = {x: probe.predict_proba(np.load(f"{C}/{x}/16.npy"))[:, 1] for x in st}
    lb = {x: pd.read_parquet(f"{C}/{x}/labels.parquet") for x in st}
    sat = max((sc[x] > 0.95).mean() for x in st)
    ok = sat < 0.09 and "below 9\\%" in tex
    print(f"  {'control share above 0.95, max over steps':<48} recomputed {sat:>9.4f}   paper   below 9%   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append(f"control saturation {sat:.3f}; paper says below 9%")
    common = set.intersection(*[set(lb[x].prompt_idx) for x in st])
    thr = float(np.quantile(sc[0], 0.5))
    fr = {x: (sc[x][lb[x].prompt_idx.isin(common).values] >= thr).mean() for x in st}
    band = max(100 * (fr[x] / fr[0] - 1) for x in st)
    check("benign flag-rate band, common prompts (%)", band, "+18.3", 0.05, tex)
    check("control prompts shared by all checkpoints", len(common), "166", 0, tex)

    # the reward probe on C_outcome's headline answers, first block, all 406
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import verify_reward_ladder as vrl  # noqa: E402
    P = os.path.join(_ROOT, "extension", "cache", "probe_cache_n500_clean406", "C_outcome_l16_pre_answer")
    meta = json.load(open(P + ".meta.json"))
    from sklearn.metrics import roc_auc_score  # noqa: E402
    y1 = vrl._first_block_labels(meta)
    s406 = probe.predict_proba(np.load(P + ".npz", allow_pickle=True)["X"])[:, 1]
    check("reward probe, all 406, first block", roc_auc_score(y1, s406), "0.951", 0.0015, tex)
    def w_in(path):
        pipe = pickle.load(open(path, "rb"))
        return pipe.steps[-1][1].coef_.ravel() / pipe.steps[0][1].scale_
    r = w_in(os.path.join(_ROOT, "extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl"))
    t = w_in(os.path.join(_ROOT, "extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl"))
    check("cosine, reward vs trace-final probe", r @ t / np.linalg.norm(r) / np.linalg.norm(t), "0.192", 0.0015, tex)

    # template score, matched headline protocol, first block, 406
    from quantify_structural_confound import template_score  # noqa: E402
    from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402
    keep = {m["prompt_idx"] for m in meta}
    AN = _re.compile(r"<answer>(.*?)</answer>", _re.S)
    def fb(x, r_):
        m = AN.search(x)
        if not m or not validate_equation(m.group(1).strip(), list(r_["nums"])):
            return 0
        v = evaluate_equation(m.group(1).strip())
        return int(v is not None and abs(v - int(r_["target"])) < 1e-5)
    # within each run: C_outcome -> runA, and C_SFT -> runB (the 2026-10-04 audit
    # found the paper comparing C_outcome with runB, across runs)
    for f, w_mean, w_r in (("eval_c_outcome_n500.json", "1.81", "+0.643"), ("eval_runA_postRL_n500.json", "3.23", "+0.068"),
                           ("eval_c_sft_n500.json", "3.24", "+0.168"), ("eval_runB_postRL_n500.json", "4.32", "+0.023")):
        rows = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
        ts = np.array([template_score(x) for i, r_ in enumerate(rows) if i in keep for x in r_["response"]])
        yy = np.array([fb(x, r_) for i, r_ in enumerate(rows) if i in keep for x in r_["response"]])
        check(f"template score, {f}", ts.mean(), w_mean, 0.005, tex)
        check(f"template r with correctness, {f}", np.corrcoef(ts, yy)[0, 1], w_r, 0.0015, tex)

    # the judge: no-answer rows at step 0, the dip without them, corrections
    J = os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag")
    D = {x: [json.loads(l) for l in open(f"{J}/step_{x}.jsonl")] for x in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)}
    na = {x: sum(r_["equation"] is None for r_ in D[x]) for x in D}
    check("judge no-answer rows at step 0", na[0], "108", 0, tex)
    ok = max(v for x, v in na.items() if x) == 6 and "at most 6" in tex
    print(f"  {'judge no-answer rows, max at later steps':<48} recomputed {max(v for x, v in na.items() if x):>9}   paper  at most 6   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append("judge no-answer count at later steps disagrees with 'at most 6'")
    def ar(x, drop):
        rr = [r_ for r_ in D[x] if not (drop and r_["equation"] is None)]
        return np.array([r_["correct"] for r_ in rr]), np.array([r_["judge_score"] for r_ in rr])
    check("judge step-0 AUROC without no-answer rows", roc_auc_score(*ar(0, True)), "0.769", 0.0015, tex,
          ctx=r"step-0 AUROC is {n}")
    check("judge dip without no-answer rows", roc_auc_score(*ar(30, True)) - roc_auc_score(*ar(0, True)), "-0.015", 0.0015, tex)
    ok = sum(r_["judge_score"] == 0.0 for x in D for r_ in D[x]) == 2 and "Two of the 35{,}728" in tex
    print(f"  {'judge rows scoring exactly 0':<48} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append("judge zero-score count disagrees with Appendix A")
    mc = json.load(open(os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag_n2000.json")))["multiple_comparisons"]
    want = {"AUROC 0-30": ("holm", True), "flag 0-30": ("holm", True), "flag 0-20": ("holm", False),
            "flag 0-40": ("bonferroni", True), "flag 0-30 ": ("bonferroni", False),
            "AUROC 0-30 ": ("bonferroni", True)}
    for name, (rule, val) in want.items():
        got = mc[name.strip()][rule]
        print(f"  {'correction: ' + name.strip() + ' survives ' + rule:<48} recomputed {got!s:>9}   paper {val!s:>14}   {'OK' if got == val else 'MISMATCH'}")
        if got != val:
            bad.append(f"{name.strip()} under {rule}: computed {got}, paper says {val}")

    # arm B: output form and the reward's weights
    sys.path.insert(0, os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe"))
    import surface_residual_probe as srp  # noqa: E402
    b = srp.load_any(os.path.join(_ROOT, "followup/experiments/fragility/residual_probe/probe_surface_only.pkl"))
    w = dict(zip(b.keys, np.asarray(b.scorer.w).ravel()))
    check("arm B weight tail_len", w["tail_len"], "+1.19", 0.005, tex)
    check("arm B weight ans_len", w["ans_len"], "+0.62", 0.005, tex)
    check("arm B weight n_equals", w["n_equals"], "-1.11", 0.005, tex)
    rows = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_armB_surface_step100.json")) if l.strip()]
    blocks = [AN.search(x) for i, r_ in enumerate(rows) if i in keep for x in r_["response"]]
    has = [m for m in blocks if m]
    check("arm B answers with an <answer> block", len(has), "3{,}244", 0, tex)
    check("arm B blocks that are arithmetic (%)",
          100 * sum(evaluate_equation(m.group(1).strip()) is not None for m in has) / len(has), "0.12", 0.005, tex)
    # contribution 1 restates A - B rounded
    sys.path.insert(0, os.path.join(_ROOT, "extension", "probe"))
    import verify_residual_arms as arms_gate  # noqa: E402
    kp = arms_gate.clean_prompts()
    d, lo, hi = arms_gate.paired_bootstrap(arms_gate.load_arm("eval_armA_residual_step100.json", kp),
                                           arms_gate.load_arm("eval_armB_surface_step100.json", kp), 10000, 0)
    check("arm A - arm B, pp (contribution 1)", d, "16.8", 0.05, tex)
    dR, loR, hiR = arms_gate.paired_bootstrap(arms_gate.load_arm("eval_armA_residual_step100.json", kp),
                                              arms_gate.load_arm("eval_armRaw_step100.json", kp), 10000, 0)
    check("arm A - arm R, pp (contribution 1)", dR, "11.6", 0.05, tex)
    ok = abs(loR - 9.6) < 0.1 and abs(hiR - 13.8) < 0.1 and "[9.6, 13.8]" in tex
    print(f"  {'arm A - arm R CI (contribution 1)':<48} recomputed [{loR:.1f}, {hiR:.1f}]   paper  [9.6, 13.8]   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append(f"A - R CI [{loR:.2f}, {hiR:.2f}] vs paper [9.6, 13.8]")
    import surface_residual_probe as _srp  # noqa: E402
    gap = (json.load(open(os.path.join(_ROOT, "followup/experiments/fragility/residual_probe/probe_raw_arm_recipe.pkl.meta.json")))["report"]["auroc_heldout"]
           - json.load(open(os.path.join(_ROOT, "followup/experiments/fragility/residual_probe/probe_surface_residual_l16.pkl.meta.json")))["report"]["auroc_residual_heldout"])
    check("AUROC lost to residualisation", gap, "0.144", 0.0015, tex)
    ok = abs(lo - 14.7) < 0.1 and abs(hi - 19.0) < 0.1 and "[14.7, 19.0]" in tex
    print(f"  {'arm A - arm B CI (contribution 1)':<48} recomputed [{lo:.1f}, {hi:.1f}]   paper [14.7, 19.0]   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append(f"A - B CI [{lo:.2f}, {hi:.2f}] vs paper [14.7, 19.0]")

    # small facts in the setup and appendices
    for f, w_ in (("eval_c_sft_n500.json", "0.290"),):
        rows = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
        acc = np.mean([fb(x, r_) for i, r_ in enumerate(rows) if i in keep for x in r_["response"]])
        check("C_SFT first-block, headline", acc, w_, 0.0015, tex)
    lr = lambda yv, pv: pv[yv == 1].mean() / pv[yv == 0].mean()
    y0 = pd.read_parquet(os.path.join(ACTS, "0", "labels.parquet"))["last_block"].values
    y4 = pd.read_parquet(os.path.join(ACTS, "40", "labels.parquet"))["last_block"].values
    check("LR+ change steps 0-40 (%)", 100 * (lr(y4, lad[40] >= t0) / lr(y0, lad[0] >= t0) - 1), "-31.6", 0.05, tex)
    n_scored = {x: len(pd.read_parquet(os.path.join(ACTS, str(x), "labels.parquet"))) for x in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)}
    check("ladder answers dropped at step 0", 3248 - n_scored[0], "32", 0, tex)
    ok = max(3248 - v for x, v in n_scored.items() if x) == 8 and "at most 8" in tex
    print(f"  {'ladder answers dropped, max at other steps':<48} recomputed {max(3248 - v for x, v in n_scored.items() if x):>9}   paper   at most 8   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append("ladder dropped-answer count disagrees with 'at most 8'")


def second_audit_numbers(tex):
    """Every number the 2026-10-04 audit added. Each replaces a claim the paper
    got wrong: what the arms' rewards scored on the model they actually read,
    what arm A's reward is made of, the probe's LR+ moving before its AUROC, the
    fixed-text arm behind "recalibration", and smaller facts."""
    print("\n=== numbers introduced by the second audit ===")
    import hashlib, pickle
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe"))
    sys.path.insert(0, os.path.join(_ROOT, "followup", "experiments", "fragility", "phase0_replicate"))
    import surface_residual_probe as srp  # noqa: E402
    from changepoint_lag import load_checkpoint  # noqa: E402
    from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402
    AN = re.compile(r"<answer>(.*?)</answer>", re.S)
    R = os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe")
    arm_a = srp.load_any(os.path.join(R, "probe_surface_residual_l16.pkl"))
    arm_r = srp.load_any(os.path.join(R, "probe_raw_arm_recipe.pkl"))
    arm_b = srp.load_any(os.path.join(R, "probe_surface_only.pkl"))
    reward = pickle.load(open(os.path.join(_ROOT, "extension", "cache", "steering",
                                           "probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl"), "rb"))

    def heldout(model, evalfile):
        """The arms' held-out half (sha256 of the prompt index even), first-block labels."""
        P = os.path.join(_ROOT, "extension", "cache", "probe_cache_n500_clean406", f"{model}_l16_pre_answer")
        X = np.load(P + ".npz", allow_pickle=True)["X"]
        meta = json.load(open(P + ".meta.json"))
        rows = [json.loads(l) for l in open(os.path.join(_ROOT, evalfile)) if l.strip()]
        texts, y = [], []
        for m in meta:
            r_ = rows[m["prompt_idx"]]
            t = r_["response"][m["resp_idx"]]
            mm = AN.search(t)
            ok = 0
            if mm and validate_equation(mm.group(1).strip(), list(r_["nums"])):
                v = evaluate_equation(mm.group(1).strip())
                ok = int(v is not None and abs(v - int(r_["target"])) < 1e-5)
            texts.append(t)
            y.append(ok)
        te = np.array([int(hashlib.sha256(str(int(m["prompt_idx"])).encode()).hexdigest(), 16) % 2 == 0
                       for m in meta])
        return X[te], [texts[i] for i in np.where(te)[0]], np.array(y)[te]

    # Table 2 reads every probe through C_outcome's hidden states. Recompute arms
    # A and B from their artefacts, not from the AUROCs their metadata recorded.
    Xo, to, yo = heldout("C_outcome", "eval_c_outcome_n500.json")
    sa = arm_a.predict_proba(Xo, text=to)[:, 1]
    check("arm A AUROC from its artefact (Table 2)", roc_auc_score(yo, sa), "0.834", 0.0015, tex)
    check("arm B AUROC from its artefact (Table 2)", roc_auc_score(yo, arm_b.predict_proba(Xo, text=to)[:, 1]),
          "0.925", 0.0015, tex)
    # During RL the arms' probes read a frozen C_SFT, on C_SFT's own answers.
    Xs, ts, ys = heldout("C_SFT", "eval_c_sft_n500.json")
    # (answers with a locatable </think>, as the paper says; during RL an answer
    # without one got reward 0, which these values do not include)
    for name, model, w in (("arm A", arm_a, "0.589"), ("arm B", arm_b, "0.755"), ("arm R", arm_r, "0.816")):
        check(f"{name} AUROC through C_SFT, on C_SFT's answers", roc_auc_score(ys, model.predict_proba(Xs, text=ts)[:, 1]),
              w, 0.0015, tex, ctx=name[-1] + r" {n}")

    # Arm A's reward is LR(h - s B): an activation term plus a fixed term in the
    # 39 text features, w.(h - mu)/sd - w.(s_z B)/sd. Decompose it.
    wx = arm_a.pipe.w / arm_a.pipe.sd
    k = len(arm_a.keys)
    tcoef = -(arm_a.B[:k] @ wx)                      # logit weight per z-scored text feature
    Sz = (srp.feature_matrix(to, arm_a.keys) - arm_a.mu) / arm_a.sigma
    act = ((Xo - arm_a.pipe.mu) / arm_a.pipe.sd) @ arm_a.pipe.w
    txt = Sz @ tcoef
    if abs(roc_auc_score(yo, act + txt) - roc_auc_score(yo, sa)) > 1e-6:
        bad.append("arm A decomposition does not reproduce arm A's own scores")
    check("arm A activation term alone, AUROC", roc_auc_score(yo, act), "0.976", 0.0015, tex)
    check("arm A text term alone, AUROC", roc_auc_score(yo, txt), "0.079", 0.0015, tex)
    bw = np.asarray(arm_b.scorer.w).ravel() / np.asarray(arm_b.scorer.sd).ravel() * arm_a.sigma
    check("corr(arm A text term, arm B weights)", np.corrcoef(tcoef, bw)[0, 1], "-0.56", 0.005, tex)
    varies = (np.abs(tcoef) > 1e-12) | (np.abs(bw) > 1e-12)        # 12 features are constant on the fit
    check("features that vary", int(varies.sum()), "27", 0, tex, ctx=r"of the {n} features that vary")
    check("of those, text term opposes arm B's sign", int((np.sign(tcoef[varies]) != np.sign(bw[varies])).sum()),
          "24", 0, tex, ctx=r"on {n} of the 27")
    keep = {m["prompt_idx"] for m in json.load(open(CLEAN))}
    def text_term(f):
        rows = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
        tx = [x for i, r_ in enumerate(rows) if i in keep for x in r_["response"]]
        return float((((srp.feature_matrix(tx, arm_a.keys) - arm_a.mu) / arm_a.sigma) @ tcoef).mean())
    check("text term on arm B's final answers (logits)", text_term("eval_armB_surface_step100.json"), "-27.9", 0.05, tex)
    check("text term on C_SFT's answers (logits)", text_term("eval_c_sft_n500.json"), "+0.16", 0.005, tex,
          ctx=r"against {n} on")

    # The probe's LR+ at the threshold frozen at step 0, paired prompt-clustered
    # bootstrap against step 0 (2,000 resamples, seed 0), under each label rule.
    L = {}
    for st in (0, 10, 30):
        sc, _, g = load_checkpoint("phase0_harvest_runA", st, 16)
        L[st] = (sc, pd.read_parquet(os.path.join(ACTS, str(st), "labels.parquet")), g)
    thr = float(np.quantile(L[0][0], 0.5))
    prompts = np.unique(L[0][2])
    idx = {st: {q: np.flatnonzero(L[st][2] == q) for q in prompts} for st in L}

    def lrp(sc, y):
        m = sc >= thr
        return m[y == 1].mean() / m[y == 0].mean()

    for rule, st, w0, ws, wp in (("first_block", 10, "7.00", "5.63", "0.010"),
                                 ("last_block", 30, "2.41", "1.91", None)):
        y0, yt = L[0][1][rule].to_numpy().astype(int), L[st][1][rule].to_numpy().astype(int)
        check(f"probe LR+ at step 0, {rule}", lrp(L[0][0], y0), w0, 0.005, tex, ctx=r"{n} \\to " + re.escape(ws))
        check(f"probe LR+ at step {st}, {rule}", lrp(L[st][0], yt), ws, 0.005, tex, ctx=re.escape(w0) + r" \\to {n}")
        if rule == "first_block":
            m0, m1 = L[0][0] >= thr, L[st][0] >= thr
            want = (f"FPR {m0[y0 == 0].mean():.3f} \\to {m1[yt == 0].mean():.3f}, "
                    f"TPR {m0[y0 == 1].mean():.3f} \\to {m1[yt == 1].mean():.3f}")
            ok = want in tex
            print(f"  {'TPR and FPR behind the LR+ fall':<48} {want}   {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append(f"TPR/FPR sentence should read '{want}'")
        rng = np.random.default_rng(0)
        d = np.empty(2000)
        for i in range(2000):
            draw = rng.choice(prompts, len(prompts))
            s0 = np.concatenate([idx[0][q] for q in draw])
            s1 = np.concatenate([idx[st][q] for q in draw])
            d[i] = lrp(L[st][0][s1], yt[s1]) - lrp(L[0][0][s0], y0[s0])
        pv = 2 * min((d >= 0).mean(), (d <= 0).mean())
        if wp is None:
            ok = pv < 0.001 and re.search(re.escape(ws) + r" by step 30 under last-block labels \(p<0\.001\)", tex) is not None
            print(f"  {'probe LR+ p, step ' + str(st) + ', ' + rule:<48} recomputed {pv:>9.4f}   paper         <0.001   {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append(f"LR+ step {st} {rule}: p {pv:.4f}, paper says < 0.001")
        else:
            # 2,000 draws carry a Monte Carlo error of about 0.002 at p = 0.01,
            # so the paper writes p ~ 0.01 and this checks that, not three decimals
            ok = abs(pv - 0.01) <= 0.004 and "p\\approx 0.01" in tex
            print(f"  {'probe LR+ p, step ' + str(st) + ', ' + rule:<48} recomputed {pv:>9.4f}   paper     p~0.01   {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append(f"LR+ step {st} {rule}: p {pv:.4f}, paper says p ~ 0.01")

    # The fixed-text arm: the step-0 answers forward-passed through checkpoint 40.
    F = os.path.join(_ROOT, "followup", "acts", "phase0_harvest_runA__fixed_text", "40")
    sf = reward.predict_proba(np.load(os.path.join(F, "16.npy")))[:, 1]
    yf = pd.read_parquet(os.path.join(F, "labels.parquet"))["first_block"].to_numpy().astype(int)
    if len(sf) != len(L[0][0]):
        bad.append("fixed-text arm does not hold the step-0 answers")
    check("fixed text: step-0 answers' mean score at step 40", sf.mean(), "0.377", 0.0015, tex, ctx=r"{n} against 0\.473")
    check("fixed text: first-block AUROC at step 40", roc_auc_score(yf, sf), "0.913", 0.0015, tex, ctx=r"{n} against 0\.919")

    # Arm R's probe against the trace-final probe, in input space as for 0.192.
    tf = pickle.load(open(os.path.join(_ROOT, "extension", "cache", "steering",
                                       "probe_pipeline_C_outcome_l16_pre_answer.pkl"), "rb"))
    wt = tf.steps[-1][1].coef_.ravel() / tf.steps[0][1].scale_
    wr = arm_r.pipe.w / arm_r.pipe.sd
    check("cosine, arm R probe vs trace-final probe", wr @ wt / np.linalg.norm(wr) / np.linalg.norm(wt), "0.779", 0.0015, tex, ctx=r"cosine {n}")

    # The control ladder's step-30 answers with no locatable </think>.
    clean = {m["prompt_idx"] for m in json.load(open(CLEAN))}
    r30 = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_c_outcome_step_30_n200.json")) if l.strip()]
    n_all = sum(len(r30[i]["response"]) for i in range(len(r30)) if i in clean)
    n_scored = len(pd.read_parquet(os.path.join(_ROOT, "followup", "acts", "vanilla_rloo_ladder", "30", "labels.parquet")))
    check("control step-30 answers unscoreable (%)", 100 * (1 - n_scored / n_all), "32.1", 0.05, tex)

    # How fixed the judge is, beyond the threshold: repeats of a (problem, equation) pair.
    key = {}
    for st in STEPS:
        for l in open(os.path.join(JUDGE, f"step_{st}.jsonl")):
            r_ = json.loads(l)
            if r_["equation"]:
                key.setdefault((r_["prompt_idx"], r_["equation"]), []).append(r_["judge_score"])
    spread = np.array([max(v) - min(v) for v in key.values() if len(v) > 1])
    check("judge pairs scoring identically every time (%)", 100 * (spread == 0).mean(), "79", 0.5, tex,
          ctx=r"{n}\\% score identically every time")
    check("judge pairs differing by more than 0.01", int((spread > 0.01).sum()), "109", 0, tex,
          ctx=r"{n} differ by more than 0\.01")
    check("judge largest difference between repeats", spread.max(), "0.109", 0.0015, tex, ctx=r"the largest by {n}")


def front_matter(tex):
    print("\n=== camera-ready front matter ===")
    must = {
        r"\usepackage[final]{neurips_workshop}": "final-mode style",
        "Abraham Yeung": "first author",
        "Anagha Ramaswamy": "second author",
        "InterpScience": "workshop named",
        "https://github.com/Abraham-y/reader-writer-probe-rl": "code link",
        # Modal sponsored the compute on condition of being acknowledged as a
        # funding source, so its removal must fail the gate, not slip through.
        "\\begin{ack}": "acknowledgments section",
        "compute credits from Modal": "Modal funding acknowledgment",
        "pdftitle=": "PDF title metadata",
        "https://huggingface.co/datasets/prismane16/monitor-accuracy-under-rl": "artifact link",
    }
    never = {
        "Anonymous Author": "blind-review author block",
        "Do not distribute": "submission notice",
        "on acceptance": "blind-review release promise",
        "deanonymise": "blind-review wording",
        "Under review": "preprint notice",
    }
    for s, what in must.items():
        ok = s in tex
        print(f"  {what:<48} {'present' if ok else 'MISSING'}")
        if not ok:
            bad.append(f"camera-ready is missing the {what} ({s})")
    for s, what in never.items():
        if s in tex:
            print(f"  {what:<48} STILL PRESENT")
            bad.append(f"camera-ready still contains the {what} ({s!r})")
    todos = re.findall(r"CAMERA-READY TODO:([^\n]*)", tex)
    for t in todos:
        print(f"  OPEN ITEM:{t[:110]}")
        bad.append("open camera-ready item:" + t[:160])
    if not todos:
        print("  no open camera-ready items")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    a = ap.parse_args()
    tex = open(os.path.join(_ROOT, a.tex)).read()
    # the gated numbers are written in TeX math; compare against the text a
    # reader sees, with the minus signs and spacing normalised
    flat = (tex.replace("$", "").replace("{-}", "-").replace("{+}", "+")
               .replace("{=}", "=").replace("\\,", ""))

    front_matter(tex)
    accuracy_lead(flat)
    disjoint(tex)
    protocols(flat)
    revision_numbers(flat)
    second_audit_numbers(flat)

    if skipped:
        print("\nskipped: " + "; ".join(skipped))
    if bad:
        print(f"\nFAIL ({len(bad)})")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("\nCamera-ready verified.")


if __name__ == "__main__":
    main()
