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
    ok_val = tol is None or abs(got - float(written.replace("+", ""))) <= tol
    ok_tex = written in tex
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
                     "eval_runB_armprotocol_step100.json"]}
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
    check("arm A first-block, arm protocol", acc["eval_armA_residual_step100.json"], "0.1678", 0.00015, tex)
    check("arm B first-block, arm protocol", acc["eval_armB_surface_step100.json"], "0.0000", 0.00015, tex)
    # Table 2 puts all three arms on the arm protocol. Recompute its two contrasts
    # and the protocol's own effect on arm C with the arms gate's estimator.
    sys.path.insert(0, os.path.join(_ROOT, "extension", "probe"))
    import verify_residual_arms as arms_gate  # noqa: E402
    keep = arms_gate.clean_prompts()
    per = {f: arms_gate.load_arm(f, keep) for f in
           ("eval_armA_residual_step100.json", "eval_armB_surface_step100.json",
            "eval_runB_armprotocol_step100.json", "eval_runB_postRL_n500.json")}
    c_arm = per["eval_runB_armprotocol_step100.json"]
    for label, treat, ref, w_pt, w_ci in (
            ("arm A - arm C (Table 2)", "eval_armA_residual_step100.json", c_arm, "+8.96", "[+7.17,+10.81]"),
            ("arm B - arm C (Table 2)", "eval_armB_surface_step100.json", c_arm, "-7.82", "[-9.05,-6.62]"),
            ("arm C: arm - headline protocol", "eval_runB_armprotocol_step100.json",
             per["eval_runB_postRL_n500.json"], "+0.48", "[-0.62, +1.60]")):
        d, lo, hi = arms_gate.paired_bootstrap(per[treat], ref, 10000, 0)
        check(label + ", pp", d, w_pt, 0.006, tex)
        lo_w, hi_w = (float(x) for x in w_ci.strip("[]").split(","))
        ok = abs(lo - lo_w) < 0.1 and abs(hi - hi_w) < 0.1 and w_ci in tex
        print(f"  {label + ' CI':<48} recomputed [{lo:+.2f}, {hi:+.2f}]   paper {w_ci:>14}   "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{label} CI: recomputed [{lo:.2f}, {hi:.2f}], paper {w_ci}")


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
    flat = tex.replace("$", "").replace("{-}", "-").replace("\\,", "")

    front_matter(tex)
    accuracy_lead(flat)
    disjoint(tex)
    protocols(flat)

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
