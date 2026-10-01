#!/usr/bin/env python3
"""Gate the three read-only AUROCs the reward ladder is ranked on.

These were the paper's least-checked numbers and the ones that went wrong.
The ladder claims read-only quality orders the three rewards BACKWARDS, so it
only means anything if the three AUROCs are on one population under one label
rule. Two of them ship inside their own artifacts. The third did not exist:
the paper quoted 0.978, which is `auroc_raw_heldout` from
surface_residual_probe.py -- a raw-activation probe FRESHLY FIT inside the arms
script, not the temp-1 probe that was actually the reward in runB.
PREREGISTRATION.md mislabels that number "the published run, already have it".

So arm C is recomputed here from the shipped reward pickle, scored on the arms'
own population, restricted to the prompts the probe was never fit on -- 248 of
its 300 fitting prompts are inside the 406, which is why the restriction is not
optional. Arms A and B are read from the artifacts they were trained against.

Fails if any value drifts from the paper, or if the ordering claim inverts.
"""
from __future__ import annotations
import hashlib, json, os, pickle, re, sys, warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARMS = os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe")
CACHE = os.path.join(_ROOT, "extension", "cache")
POP = os.path.join(CACHE, "probe_cache_n500_clean406", "C_outcome_l16_pre_answer")
REWARD = os.path.join(CACHE, "steering", "probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl")
FIT = os.path.join(CACHE, "probe_cache_temp1", "C_outcome_temp1_l16_pre_answer.meta.json")


def arm_c() -> tuple[float, int, int]:
    """The reward probe, on the arms' population, on prompts it never saw."""
    d = np.load(POP + ".npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    groups = np.array([r["prompt_idx"] for r in json.load(open(POP + ".meta.json"))])
    fitted = {r["prompt_idx"] for r in json.load(open(FIT))}
    with open(REWARD, "rb") as f:
        scores = pickle.load(f).predict_proba(X)[:, 1]
    unseen = ~np.isin(groups, list(fitted))
    overlap = len(fitted & set(groups.tolist()))
    return float(roc_auc_score(y[unseen], scores[unseen])), overlap, len(set(groups[unseen]))


def main() -> None:
    tex_name = sys.argv[sys.argv.index("--tex") + 1] if "--tex" in sys.argv \
        else "writeup_interpscience.tex"
    tex = open(os.path.join(_ROOT, tex_name)).read()

    a = json.load(open(os.path.join(ARMS, "probe_surface_residual_l16.pkl.meta.json")))
    b = json.load(open(os.path.join(ARMS, "probe_surface_only.pkl.meta.json")))
    auroc_a = a["report"]["auroc_residual_heldout"]
    auroc_b = b["report"]["auroc_heldout"]
    auroc_c, overlap, n_unseen = arm_c()

    # accuracy each reward produced, as printed in the arms table
    acc = {"A": 0.1678, "B": 0.0000, "C": 0.0782}

    rows = [("A", auroc_a, "0.834", acc["A"]),
            ("C", auroc_c, "0.861", acc["C"]),
            ("B", auroc_b, "0.925", acc["B"])]

    print(f"arm C recomputed on {n_unseen} prompts the reward probe never saw "
          f"({overlap} of its 300 fitting prompts are inside the 406)")
    bad = []
    for arm, got, written, accuracy in rows:
        ok_val = abs(got - float(written)) <= 0.0015
        ok_tex = re.search(re.escape(written).replace(r"\.", r"\."), tex) is not None
        print(f"  arm {arm}: recomputed {got:.4f}  paper {written}  "
              f"acc {accuracy:.4f}  {'OK' if ok_val and ok_tex else 'MISMATCH'}")
        if not ok_val:
            bad.append(f"arm {arm}: recomputed {got:.4f}, paper says {written}")
        if not ok_tex:
            bad.append(f"arm {arm}: {written} does not appear in {tex_name}")

    # the claim itself: AUROC ascending must pair with accuracy descending
    aurocs = [r[1] for r in rows]
    accs = [r[3] for r in rows]
    if not (aurocs == sorted(aurocs) and accs == sorted(accs, reverse=True)):
        bad.append(f"ordering claim broken: AUROC {aurocs} vs accuracy {accs}")
    else:
        print("  ordering: AUROC ascending, accuracy descending -- inversion holds")

    # The per-checkpoint ladder tables carry hundreds of rates, and a flag rate
    # of 0.978 is a legitimate cell there (Table 5, seen prompts, step 60).
    # The guard is against 0.978 being quoted as a reward's AUROC, so search
    # everything except those tables.
    ladders = r"\\begin\{table\}(?:(?!\\end\{table\}).)*?\\label\{tab:(?:lag|judge|overlap)\}.*?\\end\{table\}"
    if "0.978" in re.sub(ladders, " ", tex, flags=re.S):
        bad.append("0.978 is back in the paper; it is not any reward's AUROC")

    if bad:
        print("\nFAIL")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("\nReward ladder verified.")


if __name__ == "__main__":
    main()
