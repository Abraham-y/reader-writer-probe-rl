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


def check(label, got, written, tol, tex):
    """`got` must be within tol of `written`, and `written` must be in the paper."""
    num = float(written.replace("{,}", "").replace(",", "").replace("+", ""))
    ok_val = tol is None or abs(got - num) <= tol
    # exact token: the number must not sit inside a longer number
    ok_tex = re.search(r"(?<![\d.])" + re.escape(written) + r"(?![\d])", tex) is not None
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
    for f, w_mean, w_r in (("eval_c_outcome_n500.json", "1.81", "+0.643"), ("eval_runB_postRL_n500.json", "4.32", "+0.023")):
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
    check("judge step-0 AUROC without no-answer rows", roc_auc_score(*ar(0, True)), "0.769", 0.0015, tex)
    check("judge dip without no-answer rows", roc_auc_score(*ar(30, True)) - roc_auc_score(*ar(0, True)), "-0.015", 0.0015, tex)
    ok = sum(r_["judge_score"] == 0.0 for x in D for r_ in D[x]) == 2 and "Two of the 35{,}728" in tex
    print(f"  {'judge rows scoring exactly 0':<48} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append("judge zero-score count disagrees with Appendix A")
    mc = json.load(open(os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag_n2000.json")))["multiple_comparisons"]
    want = {"AUROC 0-30": ("holm", True), "flag 0-30": ("holm", True), "flag 0-20": ("holm", False),
            "flag 0-40": ("bonferroni", True), "flag 0-30 ": ("bonferroni", False)}
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
