#!/usr/bin/env python3
"""Gate the three read-only AUROCs the reward comparison rests on.

The comparison only means anything if all three AUROCs are on ONE population
under ONE label rule. Arms A and B ship their AUROCs in their own artifacts,
computed by surface_residual_probe.py on the held-out half of the clean-406
C_outcome headline answers (prompt split by sha256 of the prompt index) with
FIRST-block labels. Arm C, the reward probe, has no such number, so it is
recomputed here on exactly those rows with exactly those labels.

HISTORY, because this gate used to get it wrong. Until 2026-10-01 arm C was
scored on the cache's own `y` -- which is the LAST-block label -- and restricted
to prompts with index >= 300, on the belief that indices 0-299 were the reward
probe's fitting prompts. They were not: the reward probe was fit on the first
300 rows of the RL training pool (eval_c_outcome_temp1_asingh_300.json), whose
indices number a different file. None of its fitting prompts are in the 406
(`fitting_overlap` below checks this by nums+target, not by index). The old gate
produced 0.861 and an A < C < B ordering; on the arms' population and label
rule arm C reads 0.945, and the ordering is A < B < C.

Fails if any AUROC drifts from the paper, or if the arm A vs arm B inversion
the paper claims stops holding.
"""
from __future__ import annotations
import hashlib, json, os, pickle, re, sys, warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402

ARMS = os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe")
CACHE = os.path.join(_ROOT, "extension", "cache")
POP = os.path.join(CACHE, "probe_cache_n500_clean406", "C_outcome_l16_pre_answer")
REWARD = os.path.join(CACHE, "steering", "probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl")
FIT_ROLLOUTS = os.path.join(_ROOT, "eval_c_outcome_temp1_asingh_300.json")
EVAL_PROMPTS = os.path.join(_ROOT, "extension", "data", "countdown_eval_500.jsonl")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)   # as test_gate.py


def _first_block_labels(meta: list[dict]) -> np.ndarray:
    """First-<answer>-block correctness, built exactly as the arms' fit built it."""
    rows = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_c_outcome_n500.json")) if l.strip()]
    y = []
    for m in meta:
        r = rows[m["prompt_idx"]]
        mm = _ANSWER.search(r["response"][m["resp_idx"]])
        ok = 0
        if mm and validate_equation(mm.group(1).strip(), list(r["nums"])):
            v = evaluate_equation(mm.group(1).strip())
            ok = int(v is not None and abs(v - int(r["target"])) < 1e-5)
        y.append(ok)
    return np.array(y)


def heldout_mask(groups: np.ndarray) -> np.ndarray:
    """The arms' held-out half: sha256(prompt index) even (surface_residual_probe.py)."""
    return np.array([int(hashlib.sha256(str(int(g)).encode()).hexdigest(), 16) % 2 == 0
                     for g in groups])


def arm_c() -> tuple[float, int]:
    """The reward probe on the arms' held-out rows, first-block labels."""
    X = np.load(POP + ".npz", allow_pickle=True)["X"]
    meta = json.load(open(POP + ".meta.json"))
    y = _first_block_labels(meta)
    te = heldout_mask(np.array([m["prompt_idx"] for m in meta]))
    with open(REWARD, "rb") as f:
        scores = pickle.load(f).predict_proba(X)[:, 1]
    return float(roc_auc_score(y[te], scores[te])), int(te.sum())


def arm_r() -> float:
    """Arm R's reward probe, recomputed from the artefact it trained against."""
    sys.path.insert(0, ARMS)
    import surface_residual_probe as srp  # noqa: E402
    X = np.load(POP + ".npz", allow_pickle=True)["X"]
    meta = json.load(open(POP + ".meta.json"))
    y = _first_block_labels(meta)
    rows = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_c_outcome_n500.json")) if l.strip()]
    texts = [rows[m["prompt_idx"]]["response"][m["resp_idx"]] for m in meta]
    te = heldout_mask(np.array([m["prompt_idx"] for m in meta]))
    probe = srp.load_any(os.path.join(ARMS, "probe_raw_arm_recipe.pkl"))
    s = probe.predict_proba(X[te], text=[texts[i] for i in np.where(te)[0]])[:, 1]
    return float(roc_auc_score(y[te], s))


def fitting_overlap() -> int:
    """Prompts shared by the reward probe's fit and the evaluation set, by content."""
    key = lambda r: (tuple(sorted(int(x) for x in r["nums"])), int(r["target"]))
    ev = {key(json.loads(l)) for l in open(EVAL_PROMPTS) if l.strip()}
    fit = [json.loads(l) for l in open(FIT_ROLLOUTS) if l.strip()]
    return sum(key(r) in ev for r in fit)


def main() -> None:
    tex_name = sys.argv[sys.argv.index("--tex") + 1] if "--tex" in sys.argv \
        else "writeup_interpscience.tex"
    tex = open(os.path.join(_ROOT, tex_name)).read()

    a = json.load(open(os.path.join(ARMS, "probe_surface_residual_l16.pkl.meta.json")))
    b = json.load(open(os.path.join(ARMS, "probe_surface_only.pkl.meta.json")))
    auroc_a = a["report"]["auroc_residual_heldout"]
    auroc_b = b["report"]["auroc_heldout"]
    auroc_c, n_rows = arm_c()
    overlap = fitting_overlap()

    # accuracy each reward produced, as printed in the arms table
    acc = {"A": 0.1678, "B": 0.0000, "C": 0.0782, "R": 0.0514}
    auroc_r = arm_r()

    rows = [("A", auroc_a, "0.834", acc["A"]),
            ("B", auroc_b, "0.925", acc["B"]),
            ("C", auroc_c, "0.945", acc["C"]),
            ("R", auroc_r, "0.978", acc["R"])]

    print(f"all three on the arms' held-out half ({n_rows} rows), first-block labels; "
          f"reward-probe fitting prompts inside the 406: {overlap}")
    bad = []
    if overlap:
        bad.append(f"{overlap} of the reward probe's fitting prompts are in the eval set")
    for arm, got, written, accuracy in rows:
        ok_val = abs(got - float(written)) <= 0.0015
        ok_tex = re.search(r"(?<![\d.])" + re.escape(written) + r"(?!\d)", tex) is not None
        print(f"  arm {arm}: recomputed {got:.4f}  paper {written}  "
              f"acc {accuracy:.4f}  {'OK' if ok_val and ok_tex else 'MISMATCH'}")
        if not ok_val:
            bad.append(f"arm {arm}: recomputed {got:.4f}, paper says {written}")
        if not ok_tex:
            bad.append(f"arm {arm}: {written} does not appear in {tex_name}")

    # the claim the paper rests on: arm B reads higher than arm A yet trained a
    # worse policy. (Arm C is not part of the claim: it reads highest and its
    # policy sits between the two, and it differs from A and B in more than the
    # reward -- see section 4.3.)
    # and on the same recipe, arm R reads higher than arm A and did worse too
    if not (auroc_r > auroc_a and acc["R"] < acc["A"]):
        bad.append(f"A/R inversion broken: AUROC A {auroc_a:.3f} R {auroc_r:.3f}, "
                   f"accuracy A {acc['A']} R {acc['R']}")
    if not (auroc_b > auroc_a and acc["B"] < acc["A"]):
        bad.append(f"A/B inversion broken: AUROC A {auroc_a:.3f} B {auroc_b:.3f}, "
                   f"accuracy A {acc['A']} B {acc['B']}")
    else:
        print("  arm B reads higher than arm A and trained a worse policy -- inversion holds")

    # A guard against "0.978" used to live here: an earlier draft quoted 0.978 as
    # runB's (arm C's) AUROC, when it was the raw probe fit inside the arms'
    # script. That raw probe is now arm R, so 0.978 is legitimately its AUROC, and
    # arm C's own value (0.945) is gated above. The guard is retired.

    if bad:
        print("\nFAIL")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("\nReward ladder verified.")


if __name__ == "__main__":
    main()
