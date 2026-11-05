#!/usr/bin/env python3
"""Gate the front matter and the structure of the InterpScience camera-ready.

Its numbers moved to verify_claims.py, verify_tables.py and verify_prose_numbers.py
on 2026-11-05, where every check pins its number in context and verify_numbers.py
fails on any number left unclaimed. What stays here has no number to claim.

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
CLEAN = os.path.join(_ROOT, "extension", "cache", "probe_cache_n500_clean406",
                     "C_outcome_l16_pre_answer.meta.json")
STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

bad: list[str] = []
skipped: list[str] = []
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import paper_ledger as L  # noqa: E402


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

def rollout_identity(tex):
    """Answers per prompt in each protocol's files, and the re-score identity."""
    print("\n=== rollout files: answers per prompt, and the re-scored arms ===")
    files = {16: ["eval_c_sft_n500.json", "eval_c_outcome_n500.json", "eval_runA_postRL_n500.json", "eval_runB_postRL_n500.json"],
             8: ["eval_armA_residual_step100.json", "eval_armB_surface_step100.json",
                 "eval_runB_armprotocol_step100.json", "eval_armRaw_step100.json"]}
    for want, fs in files.items():
        for f in fs:
            rows = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
            k = {len(r["response"]) for r in rows}
            ok = k == {want} and len(rows) == 500
            print(f"  {f:<44} {sorted(k)} answers per prompt, {len(rows)} prompts   {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append(f"{f}: {k} answers per prompt over {len(rows)} prompts, expected {want} over 500")
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
        print(f"  {'re-scored ' + orig.split('_')[1] + ' identical to original':<44} {same}/{total}   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{orig}: re-score reproduces {same}/{total} answers")


def front_matter(tex):
    print("\n=== camera-ready front matter ===")
    must = {
        r"\usepackage[dblblindworkshop, final]{neurips_2026}": "NeurIPS 2026 workshop style, final mode",
        r"\workshoptitle{Interpretability as a Science (InterpScience)}": "workshop title for the footer",
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
    front_matter(tex)
    disjoint(tex)
    rollout_identity(tex)
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
