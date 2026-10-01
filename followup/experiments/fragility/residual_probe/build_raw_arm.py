"""Build arm A's raw-probe counterpart: the same fit, without residualisation.

WHY THIS EXISTS
Arm A trained against `LR(h - s @ B)`, a probe fit on activations with the
39 surface features regressed out. The paper's open question is how much of
arm A's outcome is the residualisation. The arm that answers it differs from
arm A in that one respect only: same rows, same first-block labels, same prompt
split, same C and class weighting, same artefact class and reward code path,
same trainer and settings -- and B = 0, so nothing is subtracted.

`surface_residual_probe.fit` already trains that raw probe for comparison
(`auroc_raw_heldout`, 0.9779) but does not save it. Arm A's own artefact was
built ad hoc and records no inputs, so this script establishes them by
reproduction: it refits arm A from the inputs `test_gate.py` uses and refuses
to continue unless the refit matches the shipped arm A artefact array for array.
Only then is the raw probe from the same fit provably arm A's counterpart.

    python followup/experiments/fragility/residual_probe/build_raw_arm.py

Writes probe_raw_arm_recipe.pkl (+ .meta.json) next to this file.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import surface_residual_probe as srp  # noqa: E402
from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402

C = 0.05
SHIPPED = HERE / "probe_surface_residual_l16.pkl"
OUT = HERE / "probe_raw_arm_recipe.pkl"


def inputs():
    """Exactly test_gate.py's inputs: clean-406 C_outcome L16 cache, first-block labels."""
    P = ROOT / "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer"
    X = np.load(str(P) + ".npz")["X"]
    meta = json.load(open(str(P) + ".meta.json"))
    rows = [json.loads(l) for l in open(ROOT / "eval_c_outcome_n500.json") if l.strip()]
    an = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def ok(eq, t, n):
        eq = eq.strip()
        if not validate_equation(eq, list(n)):
            return 0
        r = evaluate_equation(eq)
        return int(r is not None and abs(r - int(t)) < 1e-5)

    texts, y, g = [], [], []
    for m in meta:
        r = rows[m["prompt_idx"]]
        resp = r["response"][m["resp_idx"]]
        mm = an.search(resp)
        texts.append(resp)
        y.append(ok(mm.group(1), r["target"], r["nums"]) if mm else 0)
        g.append(m["prompt_idx"])
    return X, texts, np.array(y), np.array(g)


def main() -> None:
    X, texts, y, g = inputs()
    print(f"rows {len(y)}, prompts {len(set(g))}, positive rate {y.mean():.4f}")

    # 1. Refit arm A and require it to equal the shipped artefact.
    refit, report = srp.fit(X, texts, y, g, C=C)
    shipped = srp.load_any(str(SHIPPED))
    checks = {
        "keys": refit.keys == shipped.keys,
        "mu": np.allclose(refit.mu, shipped.mu, atol=1e-12, rtol=0),
        "sigma": np.allclose(refit.sigma, shipped.sigma, atol=1e-12, rtol=0),
        "B": np.allclose(refit.B, shipped.B, atol=1e-9, rtol=0),
        "scorer w": np.allclose(refit.pipe.w, shipped.pipe.w, atol=1e-9, rtol=0),
        "scorer b": abs(refit.pipe.b - shipped.pipe.b) < 1e-9,
        "residual AUROC": abs(report["auroc_residual_heldout"]
                              - shipped.provenance["report"]["auroc_residual_heldout"]) < 1e-12,
        "raw AUROC": abs(report["auroc_raw_heldout"]
                         - shipped.provenance["report"]["auroc_raw_heldout"]) < 1e-12,
    }
    for k, v in checks.items():
        print(f"  refit == shipped arm A, {k:<15} {'OK' if v else 'MISMATCH'}")
    if not all(checks.values()):
        raise SystemExit("refit does not reproduce arm A; the inputs are not established. "
                         "Not building the raw arm.")

    # 2. The raw probe, exactly as fit() trains it: same rows, same C, no residual.
    te = np.array([int(hashlib.sha256(str(int(v)).encode()).hexdigest(), 16) % 2 == 0 for v in g])
    tr = ~te
    raw = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=C, class_weight="balanced")).fit(X[tr], y[tr])
    auroc = float(roc_auc_score(y[te], raw.predict_proba(X[te])[:, 1]))
    if abs(auroc - report["auroc_raw_heldout"]) > 1e-12:
        raise SystemExit(f"raw probe AUROC {auroc} != fit()'s raw {report['auroc_raw_heldout']}")
    scorer = srp.LinearScorer.from_pipeline(raw)

    # 3. Arm A's class and code path, with B = 0: subtracts exactly nothing.
    probe = srp.SurfaceResidualProbe(
        refit.keys, refit.mu, refit.sigma, np.zeros_like(refit.B), scorer,
        provenance={"layer": 16, "position": "</think>",
                    "arm": "raw counterpart of arm A (B = 0)",
                    "report": {"auroc_heldout": auroc, "n_rows": int(len(y)),
                               "n_train_rows": int(tr.sum()), "C": C,
                               "built_by": "build_raw_arm.py; refit of arm A verified identical"}})
    # Arm A's class casts activations to float64 before scoring (its reward path
    # does the same), while sklearn scores the float32 cache as-is; the class's
    # own comments record that this shifts scores by ~1.5e-7. So the exact check
    # is against the scorer on float64 activations -- B = 0 must subtract exactly
    # nothing -- and the sklearn check carries that documented tolerance.
    Xte = X[te]
    txt = [texts[i] for i in np.where(te)[0]]
    b = probe.predict_proba(Xte, text=txt)[:, 1]
    exact = float(np.abs(b - scorer.predict_proba(Xte.astype(np.float64))[:, 1]).max())
    a = raw.predict_proba(Xte)[:, 1]
    dev = float(np.abs(a - b).max())
    print(f"  B = 0 artefact vs its scorer on float64 activations: max |diff| {exact:.2e}")
    print(f"  B = 0 artefact vs sklearn on float32 activations:    max |diff| {dev:.2e}")
    if exact != 0.0 or dev > 1e-6:
        raise SystemExit("the B = 0 artefact does not reproduce the raw probe")
    srp.save(probe, str(OUT))
    back = srp.load_any(str(OUT))
    dev2 = float(np.abs(back.predict_proba(Xte, text=txt)[:, 1] - b).max())
    if dev2 != 0.0:
        raise SystemExit("saved artefact does not reload to the same scores")
    print(f"wrote {OUT.name}: held-out AUROC {auroc:.4f} (arm A residual: "
          f"{report['auroc_residual_heldout']:.4f}); reload max |diff| {dev2:.2e}")


if __name__ == "__main__":
    main()
