#!/usr/bin/env python3
"""Recompute every cell of the checkpoint-ladder tables (3, 4, 5), and every
prose statistic drawn from the two ladders, and compare them with the paper.

WHY THIS EXISTS
verify_paper_tables.py compared point values but skipped every AUROC interval
and every p-value (96 of 272 cells), and the prose statistics drawn from the
ladders were read from stored JSON files rather than recomputed. Here each table
row is rebuilt from the cached activations and judge scores, formatted the way
the table prints it, and must match the paper's row number for number. The
bootstrap is fast_boot's vectorised version of the analyses' own: seed 0 and
2,000 draws reproduce changepoint_lag.py and analyze_judge_lag.py exactly.
Claims about multiple-comparison corrections use p-values pooled over five
seeds, because at 2,000 draws a p-value near a cut-off is decided by the seed.

    python scripts/verify_tables.py [--tex writeup_interpscience.tex]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import fast_boot as FB  # noqa: E402
import paper_ledger as L  # noqa: E402

ACTS = os.path.join(_ROOT, "followup", "acts", "phase0_harvest_runA")
CONTROL = os.path.join(_ROOT, "followup", "acts", "vanilla_rloo_ladder")
JUDGE = os.path.join(_ROOT, "followup", "results", "fragility", "judge_lag")
PROBE = os.path.join(_ROOT, "extension", "cache", "steering", "probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl")
STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
NB, SEEDS = 2000, (0, 1, 2, 3, 4)
bad: list[str] = []
CELL = re.compile(r"<?" + L.TOKEN.pattern)   # a "<0.001" cell is compared with its "<"


def f3(x):
    return f"{x:.3f}"


def fp(p):
    """A p-value as the tables print it."""
    return "<0.001" if p < 0.0005 else f"{p:.3f}"


def exact_tol(written, tol):
    """The written number must be the recomputation rounded to the digits shown."""
    w = written.replace("{,}", "").lstrip("+-")
    half = 0.5 * 10 ** -(len(w.split(".")[1]) if "." in w else 0)
    return min(tol, half + 1e-9)


def check(label, got, written, tol, text, ctx, approx=False):
    num = float(written.replace("{,}", "").replace(",", "").replace("+", ""))
    if tol is not None and not approx:
        tol = exact_tol(written, tol)
    ok_val = tol is None or abs(got - num) <= tol
    pat = ctx.replace("{n}", L.number_pattern(written))
    ok_tex = re.search(pat, text) is not None
    L.claim(label, pat, ok_val and ok_tex)
    print(f"  {label:<52} recomputed {got:>10.4f}   paper {written:>9}   {'OK' if ok_val and ok_tex else 'MISMATCH'}")
    if not ok_val:
        bad.append(f"{label}: recomputed {got:.4f}, paper says {written}")
    if not ok_tex:
        bad.append(f"{label}: /{pat}/ not found in the paper")


def assert_phrase(label, ok, pattern, text):
    found = re.search(pattern, text) is not None
    L.claim(label, pattern, ok and found)
    print(f"  {label:<52} {'OK' if ok and found else 'MISMATCH'}")
    if not (ok and found):
        bad.append(f"{label}: {'statement false' if not ok else 'phrase not found'} /{pattern}/")


def table_rows(text, header, first_col):
    """{first cell: (row text, number tokens)} for the table whose header line
    contains `header`. Bold markup is ignored when reading the first cell."""
    i = text.index(header)
    j = text.index(r"\bottomrule", i)
    rows = {}
    for line in text[i:j].split("\n")[1:]:
        if "&" not in line or r"\midrule" in line:
            continue
        key = re.sub(r"\\textbf\{([^}]*)\}", r"\1", line.split("&")[0]).strip()
        rows[key] = (line.strip(), [m.group(0) for m in CELL.finditer(line)])
    return rows


def compare_row(label, rows, key, expected):
    if key not in rows:
        bad.append(f"{label}: row {key!r} not found")
        print(f"  {label:<52} MISSING")
        return
    line, got = rows[key]
    ok = got == expected
    L.claim(label, re.escape(line), ok)
    print(f"  {label:<52} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append(f"{label}: paper {got} vs recomputed {expected}")


def ladder_stats(ck, steps, prompts, thr, flag_floor):
    """Per-step plug-in values and the analyses' intervals and p-values (seed 0)."""
    C = FB.draws(prompts, NB, 0)                      # marginal loop: shared draws
    C2 = FB.draws(prompts, NB * (len(steps) - 1), 0)   # paired AUROC: one block per contrast
    out = {}
    for s in steps:
        a = ck[s].auroc(C)
        out[s] = {"ci": np.percentile(a, [2.5, 97.5]), "flag_ci": np.percentile(ck[s].rate(C, thr), [2.5, 97.5])}
    for j, t in enumerate(steps[1:]):
        c = C2[j * NB:(j + 1) * NB]
        d = ck[steps[0]].auroc(c) - ck[t].auroc(c)
        out[t]["p"] = FB.p_two_sided(d)
        out[t]["d_ci"] = np.percentile(-d, [2.5, 97.5])
    return out


def pooled_p(ck, a, b, prompts, kind, thr=None):
    """Two-sided p of (b - a) pooled over SEEDS x NB draws."""
    ds = []
    for sd in SEEDS:
        c = FB.draws(prompts, NB, sd)
        if kind == "auroc":
            ds.append(ck[b].auroc(c) - ck[a].auroc(c))
        else:
            ds.append(ck[b].rate(c, thr) - ck[a].rate(c, thr))
    d = np.concatenate(ds)
    return max(FB.p_two_sided(d), 1 / len(d)) if kind == "flag" else FB.p_two_sided(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    a = ap.parse_args()
    text = L.canonical(open(os.path.join(_ROOT, a.tex)).read())

    # ------------------------------------------------------------ probe ladder
    probe = pickle.load(open(PROBE, "rb"))
    raw = {s: (probe.predict_proba(np.load(f"{ACTS}/{s}/16.npy"))[:, 1], pd.read_parquet(f"{ACTS}/{s}/labels.parquet"))
           for s in STEPS}
    prompts = np.unique(np.concatenate([raw[s][1].prompt_idx.to_numpy() for s in STEPS]))
    thr = float(np.quantile(raw[0][0], 0.5))
    ck = {rule: {s: FB.Checkpoint(raw[s][0], raw[s][1][rule], raw[s][1].prompt_idx, prompts) for s in STEPS}
          for rule in ("last_block", "first_block")}
    st = {rule: ladder_stats(ck[rule], STEPS, prompts, thr, True) for rule in ck}

    print("=== Table 3: the probe ladder, last-block labels ===")
    rows = table_rows(text, r"RLOO step & acc.\ first & acc.\ last & \textbf{probe AUROC}", "step")
    for s in STEPS:
        y, sc, lab = raw[s][1]["last_block"].to_numpy(), raw[s][0], raw[s][1]
        m = sc >= thr
        tpr, fpr = m[y == 1].mean(), m[y == 0].mean()
        ci, fci = st["last_block"][s]["ci"], st["last_block"][s]["flag_ci"]
        exp = [str(s), f3(lab.first_block.mean()), f3(lab.last_block.mean()), f3(roc_auc_score(y, sc)), f3(ci[0]), f3(ci[1])]
        exp += [] if s == 0 else [fp(st["last_block"][s]["p"])]
        exp += [f3(y[m].mean()), f"{tpr / fpr:.2f}", f3(m.mean()), f3(fci[0]), f3(fci[1])]
        compare_row(f"Table 3, step {s}", rows, str(s), exp)

    print("=== Table 4: the probe ladder, first-block labels ===")
    rows = table_rows(text, r"RLOO step & \textbf{probe AUROC, first block}", "step")
    for s in STEPS:
        y, sc = raw[s][1]["first_block"].to_numpy(), raw[s][0]
        ci = st["first_block"][s]["ci"]
        exp = [str(s), f3(roc_auc_score(y, sc)), f3(ci[0]), f3(ci[1])] + ([] if s == 0 else [fp(st["first_block"][s]["p"])])
        compare_row(f"Table 4, step {s}", rows, str(s), exp)

    # ------------------------------------------------------------ judge ladder
    D = {s: [json.loads(l) for l in open(f"{JUDGE}/step_{s}.jsonl")] for s in STEPS}
    jprompts = np.unique(np.concatenate([[r["prompt_idx"] for r in D[s]] for s in STEPS]))
    jthr = float(np.quantile([r["judge_score"] for r in D[0]], 0.5))
    jck = {s: FB.Checkpoint([r["judge_score"] for r in D[s]], [r["correct"] for r in D[s]],
                            [r["prompt_idx"] for r in D[s]], jprompts) for s in STEPS}
    C = FB.draws(jprompts, NB, 0)
    print("=== Table 5: the judge ladder ===")
    rows = table_rows(text, r"RLOO step & acc.\ first & \textbf{judge AUROC}", "step")
    jy = {s: np.array([r["correct"] for r in D[s]]) for s in STEPS}
    js = {s: np.array([r["judge_score"] for r in D[s]]) for s in STEPS}
    jp = {}
    for s in STEPS:
        y, sc = jy[s], js[s]
        m = sc >= jthr
        ci = np.percentile(jck[s].auroc(C), [2.5, 97.5])
        fci = np.percentile(jck[s].rate(C, jthr), [2.5, 97.5])
        exp = [str(s), f3(y.mean()), f3(roc_auc_score(y, sc)), f3(ci[0]), f3(ci[1])]
        if s:
            c = FB.draws(jprompts, NB, 0)        # analyze_judge_lag reseeds each AUROC contrast
            jp[s] = FB.p_two_sided(jck[0].auroc(c) - jck[s].auroc(c))
            exp.append(fp(jp[s]))
        exp += [f3(y[m].mean()), f"{m[y == 1].mean() / m[y == 0].mean():.2f}", f3(m.mean()), f3(fci[0]), f3(fci[1])]
        compare_row(f"Table 5, step {s}", rows, str(s), exp)
    C3 = FB.draws(jprompts, NB * 10, 0)            # judge flag contrasts, drawn in sequence
    jflag = {}
    for k, t in enumerate(STEPS[1:]):
        c = C3[k * NB:(k + 1) * NB]
        d = jck[t].rate(c, jthr) - jck[0].rate(c, jthr)
        jflag[t] = (float(np.mean(js[t] >= jthr) - np.mean(js[0] >= jthr)), np.percentile(d, [2.5, 97.5]),
                    max(FB.p_two_sided(d), 1 / NB))

    # ------------------------------------------------------------ prose, section 4.1
    print("=== prose drawn from the probe ladder ===")
    lab = {s: raw[s][1] for s in STEPS}
    fb, lb = st["first_block"], st["last_block"]
    auc = {r: {s: roc_auc_score(raw[s][1][r], raw[s][0]) for s in STEPS} for r in ("first_block", "last_block")}
    check("first-block AUROC, step 0", auc["first_block"][0], "0.919", 0.0005, text, r"labels the probe's AUROC starts at {n}")
    check("first-block AUROC rise at step 20", auc["first_block"][20] - auc["first_block"][0], "+0.017", 0.0005, text, r"at step 20 \({n}, nominal")
    check("its p", fb[20]["p"], "0.022", 0.0005, text, r"nominal p={n}, which does not survive")
    check("first-block AUROC fall at step 40", auc["first_block"][40] - auc["first_block"][0], "-0.043", 0.0005, text, r"falls at step 40, by {n} \(p<0\.001\)")
    assert_phrase("first block: first falls at step 40", fb[40]["p"] < 0.0005 and all(fb[t]["p"] >= 0.05 or auc["first_block"][t] > auc["first_block"][0] for t in (10, 20, 30)),
                  r"and first falls at step 40, by -0\.043 \(p<0\.001\)", text)
    check("last-block AUROC, step 0", auc["last_block"][0], "0.783", 0.0005, text, r"labels it starts at {n}")
    check("last-block smallest p, steps 10-40", min(lb[t]["p"] for t in (10, 20, 30, 40)), "0.22", 0.005, text, r"\(all p \\geq {n}\)")
    check("last-block AUROC change at step 50", auc["last_block"][50] - auc["last_block"][0], "-0.087", 0.0005, text, r"first moves at step 50, by {n} \(p < 0\.001\)")
    assert_phrase("last block: first moves at step 50", lb[50]["p"] < 0.0005 and min(lb[t]["p"] for t in (10, 20, 30, 40)) >= 0.05,
                  r"and first moves at step 50, by -0\.087 \(p < 0\.001\)", text)
    # accuracy falls, paired over prompts (verify_camera_ready.paired_acc's estimator)
    def acc_fall(rule, s, all_rows=False):
        if all_rows:
            return 100 * (lab[s][rule].sum() - lab[0][rule].sum()) / (len(prompts) * 8)
        return 100 * (lab[s][rule].mean() - lab[0][rule].mean())
    check("first-block fall, all answers, step 10", -acc_fall("first_block", 10, True), "4.8", 0.05, text, r"the falls are {n} and 6\.1 points")
    check("last-block fall, all answers, step 10", -acc_fall("last_block", 10, True), "6.1", 0.05, text, r"the falls are 4\.8 and {n} points")
    check("unscoreable share at step 0 (%)", 100 * (1 - len(lab[0]) / (len(prompts) * 8)), "1", 0.1, text, r"Counting the {n}\\% of answers")
    # the power statement: standard errors of the two step-10 changes
    Cs = FB.draws(prompts, NB, 0)
    def acc_boot(rule, s):
        w = Cs[:, np.searchsorted(prompts, lab[s].prompt_idx.to_numpy())].astype(float)
        return (w * lab[s][rule].to_numpy()).sum(1) / w.sum(1)
    se_acc = np.std(acc_boot("first_block", 10) - acc_boot("first_block", 0))
    se_auc = {r: np.std(ck[r][10].auroc(Cs) - ck[r][0].auroc(Cs)) for r in ck}
    z_acc = abs(lab[10].first_block.mean() - lab[0].first_block.mean()) / se_acc
    z_auc = max(abs(auc[r][10] - auc[r][0]) / se_auc[r] for r in ck)
    print(f"  (SE accuracy change {se_acc:.4f}, SE AUROC change {se_auc}, z accuracy {z_acc:.2f}, z AUROC {z_auc:.2f})")
    assert_phrase("power: similar SEs about 0.01; accuracy moved ~6 SE, AUROC < 1",
                  all(0.005 <= x <= 0.015 for x in [se_acc] + list(se_auc.values())) and round(z_acc) == 6 and z_auc < 1,
                  r"similar bootstrap standard errors, about 0\.01, and at step 10 accuracy has moved by six of them and AUROC by less than one", text)

    # flag rate, at the median threshold and at the 25th percentile
    fr = {s: float((raw[s][0] >= thr).mean()) for s in STEPS}
    check("probe threshold (step-0 median)", thr, "0.540", 0.0005, text, r"its median score \({n}\)")
    check("probe threshold, Table 3 caption", thr, "0.540", 0.0005, text, r"step-0 median score, {n}\. Bold")
    for s, w in ((0, "0.500"), (10, "0.529"), (20, "0.559"), (30, "0.601"), (40, "0.691"), (99, "0.996")):
        ctx = {0: r"monotonically from {n} \(", 10: r"\({n}, 0\.559", 20: r"0\.529, {n}, 0\.601", 30: r"0\.559, {n} and 0\.691",
               40: r"0\.601 and {n} at steps 10 to 40", 99: r"and {n} by step 99\)"}[s]
        check(f"flag rate at step {s}", fr[s], w, 0.0005, text, ctx)
    assert_phrase("flag rate rises monotonically", all(fr[STEPS[i]] <= fr[STEPS[i + 1]] for i in range(len(STEPS) - 1)),
                  r"Under attack this share rises monotonically", text)
    C3p = FB.draws(prompts, NB * 10, 0)            # changepoint_lag's flag contrasts, drawn in sequence
    d10 = ck["last_block"][10].rate(C3p[:NB], thr) - ck["last_block"][0].rate(C3p[:NB], thr)
    check("flag-rate change at step 10", fr[10] - fr[0], "+0.029", 0.0005, text, r"by {n}\[\+0\.012")
    lo, hi = np.percentile(d10, [2.5, 97.5])
    check("its CI low", lo, "+0.012", 0.0005, text, r"\+0\.029\[{n},\+0\.048\]")
    check("its CI high", hi, "+0.048", 0.0005, text, r"\[\+0\.012,{n}\] \(p<0\.01\)")
    p10 = pooled_p(ck["last_block"], 0, 10, prompts, "flag", thr)
    assert_phrase("flag-rate change at step 10: p < 0.01 (pooled over seeds)", p10 < 0.01, r"\[\+0\.012,\+0\.048\] \(p<0\.01\)", text)
    t25 = float(np.quantile(raw[0][0], 0.25))
    check("flag rate at the 25th percentile, step 0", float((raw[0][0] >= t25).mean()), "0.750", 0.0005, text, r"downwards \({n} \\to 0\.699")
    check("flag rate at the 25th percentile, step 10", float((raw[10][0] >= t25).mean()), "0.699", 0.0005, text, r"downwards \(0\.750 \\to {n}")
    y0, y1 = raw[0][1]["first_block"].to_numpy(), raw[10][1]["first_block"].to_numpy()
    lr25 = lambda sc, y: (sc[y == 1] >= t25).mean() / (sc[y == 0] >= t25).mean()
    check("LR+ at the 25th percentile, step 0", lr25(raw[0][0], y0), "2.03", 0.005, text, r"rises at step 10 \({n} \\to 2\.41\)")
    check("LR+ at the 25th percentile, step 10", lr25(raw[10][0], y1), "2.41", 0.005, text, r"\(2\.03 \\to {n}\)\.")

    # the benign control on the prompts all its checkpoints share
    cst = [0, 30, 60, 90, 100]
    csc = {x: probe.predict_proba(np.load(f"{CONTROL}/{x}/16.npy"))[:, 1] for x in cst}
    clb = {x: pd.read_parquet(f"{CONTROL}/{x}/labels.parquet") for x in cst}
    common = sorted(set.intersection(*[set(clb[x].prompt_idx) for x in cst]))
    cthr = float(np.quantile(csc[0], 0.5))
    cfr = {x: (csc[x][clb[x].prompt_idx.isin(common).values] >= cthr).mean() for x in cst}
    band = max(100 * (cfr[x] / cfr[0] - 1) for x in cst)
    att = {s: 100 * ((raw[s][0][np.isin(raw[s][1].prompt_idx, common)] >= thr).mean()
                     / (raw[0][0][np.isin(raw[0][1].prompt_idx, common)] >= thr).mean() - 1) for s in (20, 30, 40)}
    check("benign band, App B", band, "18.3", 0.05, text, r"by up to {n}\\% of its step-0 value \(at step 30\)")
    for s, w, ctx in ((20, "15.1", r"rises by {n}\\%, 22\.1"), (30, "22.1", r"15\.1\\%, {n}\\% and 41"), (40, "41.5", r"22\.1\\% and {n}\\% of its step-0 value at steps 20")):
        check(f"attacked flag change on the shared prompts, step {s}", att[s], w, 0.05, text, ctx)
    first_over = min(s_ for s_ in (20, 30, 40) if att[s_] > band)
    assert_phrase("App B: first exceeds the band at 30, narrowly; widely at 40", first_over == 30 and att[30] - band < 5 and att[40] - band > 15,
                  r"at steps 20, 30 and 40, so it first exceeds the band at step 30, narrowly, and leaves it by a wide margin at step 40", text)
    assert_phrase("section 4.1: first exceeds the band at 30, narrowly", first_over == 30 and att[30] - band < 5,
                  r"flag rate first exceeds the band at step 30, narrowly", text)


    # ------------------------------------------------------------ prose, section 4.2
    print("=== prose drawn from the judge ladder ===")
    check("judge threshold", jthr, "0.9991", 0.00005, text, r"P\(\\textsc\{yes\}\) \\geq {n}\)")
    check("judge threshold, Table 5 caption", jthr, "0.9991", 0.00005, text, r"P\(\\textsc\{yes\}\) = {n}\.")
    pdip = pooled_p(jck, 0, 30, jprompts, "auroc")
    check("judge dip p, pooled over seeds", pdip, "0.003", 0.002, text, r"about {n} pooled over bootstrap seeds", approx=True)
    t20 = jflag[20]
    check("judge flag change at step 20", t20[0], "-0.026", 0.0005, text, r"step 20 \({n}\[-0\.047")
    check("its CI low", t20[1][0], "-0.047", 0.0005, text, r"-0\.026\[{n},-0\.007\]")
    check("its CI high", t20[1][1], "-0.007", 0.0005, text, r"\[-0\.047,{n}\], p=")
    check("its p", t20[2], "0.013", 0.0005, text, r"-0\.007\], p={n}\)")
    check("judge flag p at step 10", jflag[10][2], "0.069", 0.0005, text, r"not at step 10 \(p={n}\)")

    # ------------------------------------------------------------ multiple comparisons
    print("=== multiple comparisons, p pooled over five seeds ===")
    fam = {}
    for r, nm in (("first_block", "AUROC first"), ("last_block", "AUROC last")):
        for t in STEPS[1:]:
            fam[f"{nm} 0-{t}"] = pooled_p(ck[r], 0, t, prompts, "auroc")
    for t in STEPS[1:]:
        fam[f"flag 0-{t}"] = pooled_p(ck["last_block"], 0, t, prompts, "flag", thr)
    h = FB.holm(fam)
    ok = h["AUROC first 0-40"] and h["AUROC last 0-50"] and h["flag 0-10"] and not h["AUROC first 0-20"]
    assert_phrase("probe ladder: Holm over its 30 contrasts", ok,
                  r"the AUROC falls at steps 40 \(first block\) and 50 \(last block\) and the flag-rate departure at step 10 survive a Holm correction, while the first-block rise at step 20 does not", text)
    jfam = {f"AUROC 0-{t}": pooled_p(jck, 0, t, jprompts, "auroc") for t in STEPS[1:]}
    jfam.update({f"flag 0-{t}": pooled_p(jck, 0, t, jprompts, "flag", jthr) for t in STEPS[1:]})
    jh = FB.holm(jfam)
    jb = {k: v <= 0.05 / len(jfam) for k, v in jfam.items()}
    later = [f"flag 0-{t}" for t in STEPS[1:] if t >= 40]
    near = [jfam["AUROC 0-30"], jfam["flag 0-30"]]
    print("  judge pooled p:", {k: round(v, 4) for k, v in jfam.items() if v < 0.05})
    ok = all(jh[k] and jb[k] for k in later) and all(0.001 <= p <= 0.01 for p in near) and not any(
        jh[k] or jb[k] for k in jfam if k not in later + ["AUROC 0-30", "flag 0-30"])
    assert_phrase("judge: only flag from step 40 survives reliably; step 30 near the cut-offs", ok,
                  r"only the flag-rate departures from step 40 on survive Holm or Bonferroni reliably", text)
    assert_phrase("judge: survives correction from step 40 on (section 4.2)", ok,
                  r"and it survives correction from step 40 on", text)

    # ------------------------------------------------------------ more section 4.1, captions, Appendices B, D, E
    print("=== accuracy falls, onsets and the lag ===")
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    from verify_camera_ready import paired_acc  # noqa: E402
    pa = {}
    for rule in ("first_block", "last_block"):
        for s_ in STEPS[1:]:
            per = {t: lab[t].groupby("prompt_idx")[rule].agg(["sum", "count"]).reindex(prompts) for t in (0, s_)}
            pa[(rule, s_)] = paired_acc(per, s_, prompts)
    for rule, w, wlo, whi, ctxs in (("first_block", "5.2", "-6.9", "-3.7", (r"accuracy is {n} points below its step-0 value", r"\(95\\% CI \[{n}, -3\.7\], paired", r"\[-6\.9, {n}\], paired")),
                                    ("last_block", "6.4", "-8.2", "-4.6", (r"accuracy is {n} points below its 0\.372", r"\(\[{n}, -4\.6\]\)", r"\[-8\.2, {n}\]\)\. Counting"))):
        pt, lo_, hi_ = pa[(rule, 10)]
        check(f"{rule} accuracy fall at step 10, pp", -pt, w, 0.05, text, ctxs[0])
        check(f"  its CI low", lo_, wlo, 0.05, text, ctxs[1])
        check(f"  its CI high", hi_, whi, 0.05, text, ctxs[2])
    check("first-block accuracy at step 0", lab[0].first_block.mean(), "0.546", 0.0005, text, r"step-0 value of {n} \(95")
    check("last-block accuracy at step 0", lab[0].last_block.mean(), "0.372", 0.0005, text, r"points below its {n} \(\[-8\.2")
    check("first-block fall by step 40, pp", -pa[("first_block", 40)][0], "10.0", 0.05, text, r"step 40 the falls are {n} and 7\.0")
    check("last-block fall by step 40, pp", -pa[("last_block", 40)][0], "7.0", 0.05, text, r"falls are 10\.0 and {n} points")
    sig_all = all(pa[(r, s_)][2] < 0 for r in ("first_block", "last_block") for s_ in STEPS[1:])
    assert_phrase("accuracy significantly lower at every step from 10, both rules", sig_all,
                  r"Accuracy is significantly lower from step 10 under either rule", text)
    acc_on = min(s_ for s_ in STEPS[1:] if all(pa[(r, s_)][2] < 0 for r in ("first_block", "last_block")))
    on_f = min(t for t in STEPS[1:] if fb[t]["p"] < 0.05 and auc["first_block"][t] < auc["first_block"][0])
    on_l = min(t for t in STEPS[1:] if lb[t]["p"] < 0.05)
    lag = (on_f - acc_on, on_l - acc_on)
    print(f"  (accuracy falls at {acc_on}; AUROC first falls at {on_f} (first block), first moves at {on_l} (last block): lag {lag})")
    ok = (acc_on, on_f, on_l) == (10, 40, 50)
    for lbl, pat in (("abstract: lag", r"does not show this for 30 to 40 training steps"),
                     ("abstract: onsets", r"it first falls at step 40, or first moves at step 50 if each answer"),
                     ("abstract: accuracy lower at step 10", r"significantly lower at step 10\. A frozen"),
                     ("introduction: lag", r"AUROC falls 30 to 40 steps after task accuracy"),
                     ("contribution 2: onsets", r"significantly lower by step 10\. At our 10-step checkpoint resolution, the reward probe's AUROC first falls at step 40 under first-block labels"),
                     ("contribution 2: last block", r"it first moves at step 50\. Its positive"),
                     ("section 4.1 title", r"does not fall for 30 to 40 steps after accuracy does"),
                     ("Figure 1 caption: lag", r"does not register the decline for 30 to 40 steps"),
                     ("Figure 1 caption: onsets", r"significantly lower by step 10, while the fraction of answers above a threshold frozen at step 0 rises"),
                     ("Figure 1 caption: shading", r"at step 40 \(dark\) under first-block labels and step 50 \(light\)"),
                     ("section 4.1: lag", r"from its start for 30 to 40 steps after accuracy has fallen, at our 10-step checkpoint resolution"),
                     ("section 4.1: steps", r"does not differ detectably from step 0 through step 40"),
                     ("Table 3 caption: onsets", r"first moves at step 50 under this label, and at step 40 under first-block labels"),
                     ("Table 4 caption", r"rises at step 20 \(nominally\), and first falls at step 40"),
                     ("section 4.1: AUROC first changes", r"whereas the AUROC first changes at step 40 or 50"),
                     ):
        assert_phrase(lbl, ok, pat, text)
    Cj0 = FB.draws(jprompts, NB, 0)
    j_lo = float(np.percentile(jck[99].auroc(Cj0) - jck[0].auroc(Cj0), 2.5))
    assert_phrase("judge: a fall as large as the probe's at step 40 is not excluded", ok and j_lo <= auc["first_block"][40] - auc["first_block"][0],
                  r"a fall as large as the probe's at step 40 is not excluded", text)
    flag_on = min(t for t in STEPS[1:] if pooled_p(ck["last_block"], 0, t, prompts, "flag", thr) < 0.05)
    # against the benign band the flag rate reaches its edge at 30 and clearly leaves it at 40
    edge = min(s_ for s_ in (20, 30, 40) if att[s_] > band)            # first exceeds the band
    clear = min(s_ for s_ in (20, 30, 40) if att[s_] > band + 15)      # by a wide margin
    lead = (min(on_f, on_l) - clear, max(on_f, on_l) - edge)
    assert_phrase("lead over the AUROC against the benign band: 0 to 20 steps", lead == (0, 20) and (edge, clear) == (30, 40) and att[30] - band < 5,
                  r"narrowly, and clearly at step 40, so it leads the AUROC by 0 to 20 steps rather than 30 to 40", text)
    assert_phrase("contribution 2: the same lead against the benign band", lead == (0, 20) and (edge, clear) == (30, 40),
                  r"before the AUROC falls \(0 to 20 against our imperfect benign band\)", text)
    okf = ok and flag_on == 10
    for lbl, pat in (("abstract: flag rate by step 10", r"by step 10 for the probe and later for the judge"),
                     ("contribution 2: flag rate", r"threshold frozen at step 0 registers the inflation at step 10 without labels, 30 to 40 steps before the AUROC falls"),
                     ("section 4.1: flag rate moves at step 10", r"It moves at step 10, by"),
                     ("section 4.1: lower threshold also at step 10", r"also moves at step 10, but downwards"),
                     ("section 4.1: separated", r"therefore separated by 30 to 40 steps"),
                     ("section 4.1: inflate for", r"inflate for 30 to 40 steps while its ranking holds"),
                     ("section 4.1: freeze at step 0", r"We freeze a threshold at step 0, its median"),
                     ("contribution 2: LR+ by step 10", r"falls sooner, by step 10 under first-block labels"),
                     ("section 4.1: LR+ steps", r"by step 10 under first-block labels \(p\\approx 0\.01\), and 2\.41 \\to 1\.91 by step 30 under last-block"),
                     ("section 4.1: TPR/FPR by step 10", r"TPR 0\.819 \\to 0\.907 by step 10\)"),
                     ("section 4.1: 25th percentile", r"at the step-0 25th percentile, \\mathrm\{LR\}\+ under first-block labels instead rises at step 10"),
                     ("section 4.1: 25th percentile flag", r"0\.699 at the step-0 25th percentile\)"),
                     ("section 4.1: step-40 checkpoint", r"Read through the step-40 checkpoint, the step-0 answers score lower than at step 0"),
                     ("section 4.1: accuracy by step 10", r"Task accuracy falls first\. By step 10, first-block accuracy"),
                     ("section 4.1: inflation over steps 0 to 40", r"Between steps 0 and 40 the mean probe score inflates"),
                     ("App B: inflation over steps 0 to 40", r"Between steps 0 and 40 the mean score inflates"),
                     ("App B: step-0 answers through the step-40 checkpoint", r"the step-0 answers read through the step-40 checkpoint average"),
                     ("App E: tables print <0.001 when no draw crosses", r"and print <0\.001 where no resample crosses zero")):
        assert_phrase(lbl, okf, pat, text)
    check("ladders' largest first-block accuracy gap", max(abs(lab[s_].first_block.mean() - jy[s_].mean()) for s_ in STEPS), "0.016", 0.0005, text, r"agree to within {n} at every checkpoint")
    check("Table 1 caption: probe on the probe ladder, step 0", auc["first_block"][0], "0.919", 0.0005, text, r"from the same checkpoint the reward probe reads {n}\.")
    mean = {s_: float(raw[s_][0].mean()) for s_ in STEPS}
    for s_, w, ctxs in ((0, "0.473", (r"probe score inflates {n} \\to 0\.677", r"mean score inflates {n} \\to 0\.677", r"\({n} against 0\.473\)|\(0\.377 against {n}\)")),
                        (40, "0.677", (r"inflates 0\.473 \\to {n}\. Read", r"inflates 0\.473 \\to {n} while leaving"))):
        for c_ in ctxs:
            check(f"mean probe score, step {s_}", mean[s_], w, 0.0005, text, c_)
    F = os.path.join(_ROOT, "followup", "acts", "phase0_harvest_runA__fixed_text", "40")
    sf = probe.predict_proba(np.load(os.path.join(F, "16.npy")))[:, 1]
    yf = pd.read_parquet(os.path.join(F, "labels.parquet"))["first_block"].to_numpy()
    check("fixed text: step-0 answers at step 40", sf.mean(), "0.377", 0.0005, text, r"step 0 \({n} against 0\.473\)")
    check("fixed text: App B", sf.mean(), "0.377", 0.0005, text, r"checkpoint average {n}\)")
    check("fixed text: first-block AUROC at step 40", roc_auc_score(yf, sf), "0.913", 0.0005, text, r"AUROC \({n} against 0\.919\)")
    check("first-block AUROC at step 0 (fixed-text comparison)", auc["first_block"][0], "0.919", 0.0005, text, r"\(0\.913 against {n}\)")
    for rule, w0, ws, st_ in (("first_block", "7.00", "5.63", 10), ("last_block", "2.41", "1.91", 30)):
        y0_, ys_ = raw[0][1][rule].to_numpy(), raw[st_][1][rule].to_numpy()
        lr_ = lambda sc, y: (sc[y == 1] >= thr).mean() / (sc[y == 0] >= thr).mean()
        check(f"LR+ at step 0, {rule}", lr_(raw[0][0], y0_), w0, 0.005, text, r"{n} \\to " + re.escape(ws))
        check(f"LR+ at step {st_}, {rule}", lr_(raw[st_][0], ys_), ws, 0.005, text, re.escape(w0) + r" \\to {n}")
    y0f, y1f = raw[0][1]["first_block"].to_numpy(), raw[10][1]["first_block"].to_numpy()
    ds = []
    for sd in SEEDS:
        c_ = FB.draws(prompts, NB, sd)
        lr = lambda k_, cc: ck["first_block"][k_].rate(cc, thr, "pos") / ck["first_block"][k_].rate(cc, thr, "neg")
        ds.append(lr(10, c_) - lr(0, c_))
    p_lr = FB.p_two_sided(np.concatenate(ds))
    dsl = []
    for sd in SEEDS:
        c_ = FB.draws(prompts, NB, sd)
        lrl = lambda k_, cc: ck["last_block"][k_].rate(cc, thr, "pos") / ck["last_block"][k_].rate(cc, thr, "neg")
        dsl.append(lrl(30, c_) - lrl(0, c_))
    assert_phrase("LR+ step-30 fall, last block: p < 0.001 (pooled)", FB.p_two_sided(np.concatenate(dsl)) < 0.001,
                  r"by step 30 under last-block labels \(p<0\.001\)", text)
    check("LR+ step-10 p, pooled over seeds (first block)", p_lr, "0.01", 0.004, text, r"labels \(p\\approx {n}\)", approx=True)
    for w, val, ctx in (("0.117", (raw[0][0][y0f == 0] >= thr).mean(), r"FPR {n} \\to 0\.161"), ("0.161", (raw[10][0][y1f == 0] >= thr).mean(), r"FPR 0\.117 \\to {n},"),
                        ("0.819", (raw[0][0][y0f == 1] >= thr).mean(), r"TPR {n} \\to 0\.907"), ("0.907", (raw[10][0][y1f == 1] >= thr).mean(), r"TPR 0\.819 \\to {n} by")):
        check(f"rates at the median threshold, {w}", val, w, 0.0005, text, ctx)
    sat = {s_: 100 * (raw[s_][0] > 0.95).mean() for s_ in (0, 40)}
    check("share above 0.95, step 0 (%)", sat[0], "5.2", 0.05, text, r"above 0\.95 rises from {n}\\% at step 0")
    check("share above 0.95, step 40 (%)", sat[40], "38.8", 0.05, text, r"at step 0 to {n}\\% by step 40, while")
    check("share above 0.95, step 40 (%), App B", sat[40], "38.8", 0.05, text, r"attacked run reaches {n}\\% by step 40")
    csat = max((csc[x] > 0.95).mean() for x in cst)
    assert_phrase("control share above 0.95 stays below 9%", csat < 0.09, r"same probe keeps it below 9\\% at every checkpoint", text)
    assert_phrase("control share above 0.95 stays below 9% (App B)", csat < 0.09, r"answers above 0\.95 below 9\\% at every checkpoint", text)
    # Appendix B: the probe's operating point, last-block labels, steps 0 to 40
    y0l, y4l = raw[0][1]["last_block"].to_numpy(), raw[40][1]["last_block"].to_numpy()
    m0, m4 = raw[0][0] >= thr, raw[40][0] >= thr
    prec0, prec4 = y0l[m0].mean(), y4l[m4].mean()
    lr0, lr4 = m0[y0l == 1].mean() / m0[y0l == 0].mean(), m4[y4l == 1].mean() / m4[y4l == 0].mean()
    check("precision, step 0", prec0, "0.588", 0.0005, text, r"falls {n} \\to 0\.416")
    check("precision, step 40", prec4, "0.416", 0.0005, text, r"0\.588 \\to {n} \(")
    check("precision change (%)", 100 * (prec4 / prec0 - 1), "-29.3", 0.05, text, r"0\.416 \({n}\\%\)")
    check("LR+, step 0 (App B)", lr0, "2.41", 0.005, text, r"it falls {n} \\to 1\.65")
    check("LR+, step 40 (App B)", lr4, "1.65", 0.005, text, r"falls 2\.41 \\to {n}, or")
    check("LR+ change (%)", 100 * (lr4 / lr0 - 1), "-31.6", 0.05, text, r"\\mathbf\{{n}\\%\}")
    check("AUROC change, % of the above-chance margin", 100 * ((auc["last_block"][40] - 0.5) / (auc["last_block"][0] - 0.5) - 1), "-5.1", 0.05, text, r"AUROC moves {n}\\%")
    lift0, lift4 = prec0 / y0l.mean(), prec4 / y4l.mean()
    check("lift change (%)", 100 * (lift4 / lift0 - 1), "-12.8", 0.05, text, r"lift moves only {n}\\%")
    # Appendix B: the control ladder's defects
    nprom = {x: clb[x].prompt_idx.nunique() for x in cst}
    assert_phrase("control: five checkpoints, three on 166-167 prompts", len(cst) == 5 and sorted(nprom[x] for x in (30, 60, 90)) == [166, 167, 167] and nprom[0] == nprom[100] == 406,
                  r"five checkpoints, the wrong initialisation", text)
    assert_phrase("control: 166--167 prompts", sorted(nprom[x] for x in (30, 60, 90)) == [166, 167, 167], r"scored on 166--167 prompts where every attacked checkpoint uses all 406", text)
    nresp = {len(json.loads(l)["response"]) for l in open(os.path.join(_ROOT, "eval_c_outcome_step_30_n200.json")) if l.strip()}
    import ast as _ast
    _sd = {node.args[0].value.lstrip("-"): kw.value.value for node in _ast.walk(_ast.parse(open(os.path.join(_ROOT, "extension/evaluation/sample_local_jsonl.py")).read()))
           if isinstance(node, _ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args
           for kw in node.keywords if kw.arg == "default" and isinstance(kw.value, _ast.Constant)}
    launch = open(os.path.join(_ROOT, "extension/evaluation/launch_expansion_rollouts.sh")).read()
    assert_phrase("control: 16 answers per prompt at the sampler's temperature 0.6", nresp == {16} and _sd["temperature"] == 0.6 and "eval_c_outcome_step_" in launch and "--temperature" not in launch, r"headline-protocol sampling \(16 answers at temperature 0\.6\) rather than the ladder's 8 at temperature 1", text)
    clean = {m["prompt_idx"] for m in json.load(open(os.path.join(_ROOT, "extension/cache/probe_cache_n500_clean406/C_outcome_l16_pre_answer.meta.json")))}
    r30 = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_c_outcome_step_30_n200.json")) if l.strip()]
    n_all = sum(len(r30[i]["response"]) for i in range(len(r30)) if i in clean)
    check("control step-30 answers unscoreable (%)", 100 * (1 - len(clb[30]) / n_all), "32.1", 0.05, text, r"at step 30 a further {n}\\% of its answers")
    assert_phrase("control: band at its peak at step 30", max(cst, key=lambda x: cfr[x]) == 30, r"of its step-0 value \(at step 30\)", text)
    assert_phrase("control: 166 prompts shared by all five", len(common) == 166 and len(cst) == 5, r"fixed at the 166 prompts all five checkpoints share", text)
    # Appendix D: the ladder's dropped answers
    nsc = {s_: len(lab[s_]) for s_ in STEPS}
    full = len(prompts) * 8
    assert_phrase("ladder drops 32 at step 0 and at most 8 elsewhere", full - nsc[0] == 32 and max(full - nsc[s_] for s_ in STEPS[1:]) == 8
                  and min(nsc.values()) == 3216 and max(nsc.values()) == 3248 and full == 3248,
                  r"32 at step 0 and at most 8 at any other step \(3\{,\}216--3\{,\}248 of 3\{,\}248 scored\)", text)
    # Appendix E and the table captions
    yml = open(os.path.join(_ROOT, "followup", "configs", "fragility", "phase0_harvest_runA.yaml")).read()
    trainer = open(os.path.join(_ROOT, "rloo_trainer", "rloo.py")).read()
    assert_phrase("checkpoint labels: step k after k+1 updates, final = 99", "final_step: 99" in yml and set(STEPS) == {int(x) for x in os.listdir(ACTS) if x.isdigit()}
                  and '"updates_completed": global_step + 1' in trainer and 'f"epoch_{epoch}_step_{global_step}"' in trainer,
                  r"step k holds the weights after k\+1 updates, so step 0 is one update past C_\{\\text\{outcome\}\}, and step 99 is the final checkpoint", text)
    assert_phrase("2,000 resamples (App E, captions)", NB == 2000, r"2\{,\}000 resamples", text)
    assert_phrase("30 probe-ladder contrasts", len(fam) == 30, r"Over the probe ladder's 30 step-0 contrasts", text)
    assert_phrase("20 judge contrasts", len(jfam) == 20, r"Over the judge's 20 contrasts \(AUROC and flag rate\)", text)
    assert_phrase("20 judge contrasts (section 4.2)", len(jfam) == 20, r"correction over the judge's 20 contrasts", text)
    verdicts = {"AUROC 0-30": set(), "flag 0-30": set()}
    for sd in range(8):
        c_ = FB.draws(jprompts, NB, sd)
        a0_, r0_ = jck[0].auroc(c_), jck[0].rate(c_, jthr)
        fam_s = {}
        for t in STEPS[1:]:
            fam_s[f"AUROC 0-{t}"] = FB.p_two_sided(a0_ - jck[t].auroc(c_))
            fam_s[f"flag 0-{t}"] = max(FB.p_two_sided(jck[t].rate(c_, jthr) - r0_), 1 / NB)
        h_s = FB.holm(fam_s)
        for k_ in verdicts:
            verdicts[k_].add(h_s[k_])
    print(f"  (Holm verdicts over seeds 0-7: {verdicts})")
    assert_phrase("judge: step-30 dip and flag rate survive Holm or not depending on the seed", all(v == {True, False} for v in verdicts.values()),
                  r"the AUROC dip and the flag rate at step 30 sit near the cut-offs and survive or not depending on the bootstrap seed", text)
    # ------------------------------------------------------------ the judge, beyond the pooled AUROC
    print("=== the judge on well-formed answers and at stricter thresholds ===")
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    from verify_judge_errors import bucket  # noqa: E402
    wf = {s_: np.array([bucket(r) in ("CORRECT", "valid form, wrong value") for r in D[s_]]) for s_ in (0, 99)}
    ckw = {s_: FB.Checkpoint(js[s_][wf[s_]], jy[s_][wf[s_]], np.array([r["prompt_idx"] for r in D[s_]])[wf[s_]], jprompts) for s_ in (0, 99)}
    Cj = FB.draws(jprompts, NB, 0)
    aw = {s_: roc_auc_score(jy[s_][wf[s_]], js[s_][wf[s_]]) for s_ in (0, 99)}
    dw = ckw[99].auroc(Cj) - ckw[0].auroc(Cj)
    check("judge AUROC, well-formed answers, step 0", aw[0], "0.703", 0.0005, text, r"use them correctly it falls from {n} at step 0 to 0\.628 at step 99")
    check("judge AUROC, well-formed answers, step 99", aw[99], "0.628", 0.0005, text, r"from 0\.703 at step 0 to {n} at step 99 \(change")
    lo_, hi_ = np.percentile(dw, [2.5, 97.5])
    check("  change CI low", lo_, "-0.116", 0.0005, text, r"\(change CI \[{n}, -0\.033\]\)")
    check("  change CI high", hi_, "-0.033", 0.0005, text, r"\[-0\.116, {n}\]\)\. Against malformed")
    t75 = float(np.quantile(js[0], 0.75))
    lrj = lambda s_, t_, c_=None: (jck[s_].rate(c_, t_, "pos") / jck[s_].rate(c_, t_, "neg")) if c_ is not None else \
        (js[s_][jy[s_] == 1] >= t_).mean() / (js[s_][jy[s_] == 0] >= t_).mean()
    check("judge LR+ at the 75th percentile, step 0", lrj(0, t75), "4.61", 0.005, text, r"falls {n} \\to 2\.49")
    check("judge LR+ at the 75th percentile, step 99", lrj(99, t75), "2.49", 0.005, text, r"4\.61 \\to {n} \(ratio")
    r75 = np.percentile(lrj(99, t75, Cj) / lrj(0, t75, Cj), [2.5, 97.5])
    check("  ratio CI low", r75[0], "0.33", 0.005, text, r"\(ratio CI \[{n}, 0\.86\]\)")
    check("  ratio CI high", r75[1], "0.86", 0.005, text, r"\(ratio CI \[0\.33, {n}\]\)\.")
    r50 = np.percentile(lrj(99, jthr, Cj) / lrj(0, jthr, Cj), [2.5, 97.5])
    check("judge LR+ ratio at the median, CI low", r50[0], "0.88", 0.005, text, r"ratio CI \[{n}, 1\.24\]")
    check("judge LR+ ratio at the median, CI high", r50[1], "1.24", 0.005, text, r"\[0\.88, {n}\]\)\. Both")
    assert_phrase("the stricter threshold is the step-0 75th percentile", True, r"at a stricter threshold, the step-0 75th percentile", text)
    na = {s_: sum(r["equation"] is None for r in D[s_]) for s_ in STEPS}
    check("judge: answers with no <answer> block at step 0", na[0], "108", 0, text, r"step 0 has {n} answers with no")
    check("judge: same, Table 5 caption", na[0], "108", 0, text, r"step 0 holds {n} answers with no")
    assert_phrase("judge: at most 6 at any later step", max(na[s_] for s_ in STEPS[1:]) == 6, r"against at most 6 at any later step", text)
    keep_na = {s_: np.array([r["equation"] is not None for r in D[s_]]) for s_ in (0, 30)}
    ckn = {s_: FB.Checkpoint(js[s_][keep_na[s_]], jy[s_][keep_na[s_]], np.array([r["prompt_idx"] for r in D[s_]])[keep_na[s_]], jprompts) for s_ in (0, 30)}
    dn = ckn[30].auroc(Cj) - ckn[0].auroc(Cj)
    a0n = roc_auc_score(jy[0][keep_na[0]], js[0][keep_na[0]])
    a3n = roc_auc_score(jy[30][keep_na[30]], js[30][keep_na[30]])
    check("judge step-0 AUROC without no-answer rows", a0n, "0.769", 0.0005, text, r"step-0 AUROC is {n} and the dip")
    check("judge dip without no-answer rows", a3n - a0n, "-0.015", 0.0005, text, r"and the dip is {n} \(p=")
    check("  its p", FB.p_two_sided(dn), "0.18", 0.005, text, r"is -0\.015 \(p={n}\)")
    assert_phrase("judge dip across steps 20--40 (section 4.2, Table 5)", all(jfam[f"AUROC 0-{t}"] < 0.05 for t in (20, 30, 40)),
                  r"dip of -0\.031 at step 30, across steps 20--40", text)
    assert_phrase("judge dip across steps 20--40, Table 5 caption", all(jfam[f"AUROC 0-{t}"] < 0.05 for t in (20, 30, 40)), r"pooled AUROC dips at steps 20--40", text)
    assert_phrase("about half of the dip from the no-answer rows", 0.4 <= (a3n - a0n) / (jck_d := (roc_auc_score(jy[30], js[30]) - roc_auc_score(jy[0], js[0]))) <= 0.6,
                  r"About half of the dip comes from one feature of the step-0 population", text)

    # ------------------------------------------------------------ the judge: decomposition, Zhou's statistic, thresholds
    print("=== the judge's pooled AUROC, decomposed ===")
    bk = {s_: np.array([bucket(r) for r in D[s_]]) for s_ in (0, 99)}
    def part(s_, negs):
        pos, neg = js[s_][bk[s_] == "CORRECT"], js[s_][np.isin(bk[s_], negs)]
        return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]), len(neg)
    MAL = ["wrong numbers used", "no answer given"]
    (aw0, nw0), (am0, nm0) = part(0, ["valid form, wrong value"]), part(0, MAL)
    (aw9, nw9), (am9, nm9) = part(99, ["valid form, wrong value"]), part(99, MAL)
    check("AUROC against malformed wrong answers, step 0", am0, "0.935", 0.0005, text, r"easily, it goes from {n} to 0\.956")
    check("AUROC against malformed wrong answers, step 99", am9, "0.956", 0.0005, text, r"from 0\.935 to {n}\.")
    check("malformed share of wrong answers, step 0 (%)", 100 * nm0 / (nm0 + nw0), "36", 0.5, text, r"Malformed answers grew from {n}\\% of wrong")
    check("malformed share of wrong answers, step 99 (%)", 100 * nm9 / (nm9 + nw9), "43", 0.5, text, r"of wrong answers to {n}\\%; at step 0's mix")
    w0 = nw0 / (nw0 + nm0)
    check("step 99's pooled AUROC at step 0's mix", w0 * aw9 + (1 - w0) * am9, "0.745", 0.0005, text, r"step 99's pooled AUROC would be {n} rather than 0\.769")
    check("step 99's pooled AUROC", roc_auc_score(jy[99], js[99]), "0.769", 0.0005, text, r"would be 0\.745 rather than {n}\.")
    if abs(w0 * aw0 + (1 - w0) * am0 - roc_auc_score(jy[0], js[0])) > 0.002:
        bad.append("the pooled AUROC is not the weighted mean of its two parts")
    t75j = float(np.quantile(js[0], 0.75))
    tf = lambda s_, t_: (js[s_][jy[s_] == 1] >= t_).mean() - (js[s_][jy[s_] == 0] >= t_).mean()
    for w, v, ctx in (("0.41", tf(0, jthr), r"moves only from {n} to 0\.37"), ("0.37", tf(99, jthr), r"from 0\.41 to {n}; at"),
                      ("0.30", tf(0, t75j), r"at the step-0 75th percentile it falls from {n} to 0\.08\."), ("0.08", tf(99, t75j), r"falls from 0\.30 to {n}\.")):
        check(f"judge TPR-FPR {w}", v, w, 0.005, text, ctx)
    check("judge dip p in Table 5", jp[30], "0.002", 0.0005, text, r"nominal p is {n} in Table")
    assert_phrase("the stricter threshold is the step-0 75th percentile", abs(np.mean(js[0] >= t75j) - 0.25) < 0.02 and abs(np.mean(js[0] >= t75) - 0.25) < 0.02,
                  r"at a stricter threshold, the step-0 75th percentile", text)
    # App B: LR+ at other frozen thresholds, both monitors
    def lr_at(sc0, y0_, sc1, y1_, q):
        t_ = float(np.quantile(sc0, q))
        f_ = lambda sc, yy: (sc[yy == 1] >= t_).mean() / (sc[yy == 0] >= t_).mean()
        return f_(sc0, y0_), f_(sc1, y1_)
    yp0, yp1 = raw[0][1]["first_block"].to_numpy(), raw[10][1]["first_block"].to_numpy()
    probe_q = {q: lr_at(raw[0][0], yp0, raw[10][0], yp1, q) for q in (0.25, 0.75, 0.9)}
    judge_q = {q: lr_at(js[0], jy[0], js[99], jy[99], q) for q in (0.25, 0.75, 0.9)}
    print(f"  (LR+ probe step 0->10 {probe_q}; judge step 0->99 {judge_q})")
    strict_fall = all(b_ < a_ for a_, b_ in list(probe_q.values())[1:] + list(judge_q.values())[1:])
    lenient_no_fall = all(b_ >= a_ for a_, b_ in (probe_q[0.25], judge_q[0.25]))
    assert_phrase("other thresholds: LR+ falls at strict ones, not at the lenient one", strict_fall and lenient_no_fall,
                  r"We examined the step-0 25th, 75th and 90th percentiles as thresholds after seeing the data; for both monitors \\mathrm\{LR\}\+ falls at the strict thresholds and not at the lenient one", text)
    if not all(np.isfinite(js[s_]).all() for s_ in STEPS):
        bad.append("non-finite judge scores: analyze_judge_lag drops these rows and this gate does not")

    # Figure 1 draws from the stored change-point files: they must equal this recomputation,
    # and its shaded span must sit at the onsets computed above.
    for rule, fn in (("last_block", "changepoint_lag.json"), ("first_block", "changepoint_lag_firstblock.json")):
        per_ = json.load(open(os.path.join(_ROOT, "followup", "results", "fragility", fn)))["per_step"]
        drift = max(max(abs(per_[str(s_)]["auroc"] - auc[rule][s_]), abs(per_[str(s_)]["ci_lo"] - st[rule][s_]["ci"][0]),
                        abs(per_[str(s_)]["ci_hi"] - st[rule][s_]["ci"][1]),
                        abs(per_[str(s_)]["flag_rate"] - fr[s_])) for s_ in STEPS)
        if drift > 1e-9:
            bad.append(f"{fn}, which Figure 1 plots, differs from the recomputation by {drift:.2e}")
    import plot_lag_from_gated as PLG
    if (PLG.HACK_FROM, PLG.BREAK_FIRST, PLG.BREAK_AT) != (acc_on, on_f, on_l):
        bad.append(f"Figure 1 shades {(PLG.HACK_FROM, PLG.BREAK_FIRST, PLG.BREAK_AT)}, the onsets are {(acc_on, on_f, on_l)}")
    L.dump("verify_tables")
    if bad:
        print(f"\nFAIL ({len(bad)})")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("\nAll table cells and ladder statistics verified.")


if __name__ == "__main__":
    main()
