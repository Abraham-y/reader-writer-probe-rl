#!/usr/bin/env python3
"""Every number in the paper outside the checkpoint ladders, recomputed and
pinned to its sentence.

WHY THIS EXISTS
The ladder statistics are verify_tables.py's and the judge's row-level facts are
verify_prose_numbers.py's. Everything else lives here: the headline accuracies,
the arms and Table 2, Table 1, the probe baselines of section 3 and Appendix D,
the template score, the run settings of Appendix C, and Appendix F's quoted
answer. Each check recomputes its value from the cached artifacts (or, for run
settings, from the committed W&B snapshot docs/run_configs.json and the sampler
defaults in code) and must find it in the paper in its own context, so the same
digits elsewhere cannot satisfy it; verify_numbers.py then fails on any number
no check claims.

    python scripts/verify_claims.py [--tex writeup_interpscience.tex]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "extension", "probe"),
          os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe")):
    sys.path.insert(0, p)
import paper_ledger as L  # noqa: E402
import surface_residual_probe as srp  # noqa: E402
import verify_residual_arms as arms_gate  # noqa: E402
import verify_reward_ladder as vrl  # noqa: E402
from evaluation.countdown import evaluate_equation, validate_equation  # noqa: E402
from quantify_structural_confound import template_score  # noqa: E402
from verify_tables import compare_row, table_rows  # noqa: E402

ARMS = os.path.join(_ROOT, "followup", "experiments", "fragility", "residual_probe")
CLEAN = os.path.join(_ROOT, "extension", "cache", "probe_cache_n500_clean406", "C_outcome_l16_pre_answer.meta.json")
SB_JSON = os.environ.get("SB_JSON", "/tmp/_sb.json")
AN = re.compile(r"<answer>(.*?)</answer>", re.S)
bad: list[str] = []


def check(label, got, written, tol, text, ctx, approx=False):
    num = float(written.replace("{,}", "").replace(",", "").replace("+", ""))
    if not approx:
        w = written.replace("{,}", "").lstrip("+-")
        tol = min(tol, 0.5 * 10 ** -(len(w.split(".")[1]) if "." in w else 0) + 1e-9)
    ok_val = abs(got - num) <= tol
    pat = ctx.replace("{n}", L.number_pattern(written))
    ok_tex = re.search(pat, text) is not None
    L.claim(label, pat, ok_val and ok_tex)
    print(f"  {label:<52} recomputed {got:>10.4f}   paper {written:>9}   {'OK' if ok_val and ok_tex else 'MISMATCH'}")
    if not ok_val:
        bad.append(f"{label}: recomputed {got:.4f}, paper says {written}")
    if not ok_tex:
        bad.append(f"{label}: /{pat}/ not found in the paper")


def phrase(label, ok, pattern, text, setting=False):
    found = re.search(pattern, text) is not None
    L.claim(label, pattern, ok and found, setting)
    print(f"  {label:<52} {'OK' if ok and found else 'MISMATCH'}")
    if not (ok and found):
        bad.append(f"{label}: {'statement false' if not ok else 'phrase not found'} /{pattern}/")


def first_block(resp, nums, tgt):
    mm = AN.search(resp)
    if not mm or not validate_equation(mm.group(1).strip(), list(nums)):
        return 0
    v = evaluate_equation(mm.group(1).strip())
    return int(v is not None and abs(v - int(tgt)) < 1e-5)


def headline(f, keep):
    rows = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
    return float(np.mean([first_block(x, r["nums"], r["target"]) for i, r in enumerate(rows) if i in keep for x in r["response"]]))


def argparse_defaults(path):
    """{flag: default} from a script's argparse add_argument calls, read without running it."""
    out = {}
    for node in ast.walk(ast.parse(open(os.path.join(_ROOT, path)).read())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    out[node.args[0].value.lstrip("-")] = kw.value.value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    a = ap.parse_args()
    text = L.canonical(open(os.path.join(_ROOT, a.tex)).read())
    keep = {m["prompt_idx"] for m in json.load(open(CLEAN))}
    snap = json.load(open(os.path.join(_ROOT, "docs", "run_configs.json")))["runs"]
    t1 = json.load(open(os.path.join(_ROOT, "extension/cache/steering/probe_pipeline_temp1_meta.json")))["sampling_regime"].startswith("temperature=1.0")
    if not os.path.exists(SB_JSON):
        subprocess.run([sys.executable, "-W", "ignore", os.path.join(_ROOT, "extension/probe/structural_baselines.py"),
                        "--out", SB_JSON.replace(".json", ".txt")], check=True, capture_output=True)
    sb = {r["ckpt"]: r for r in json.load(open(SB_JSON))}
    so, ss = sb["C_outcome"], sb["C_SFT"]

    # ---------------------------------------------------------------- settings used throughout
    print("=== settings ===")
    prompts500 = [json.loads(l) for l in open(os.path.join(_ROOT, "extension", "data", "countdown_eval_500.jsonl")) if l.strip()]
    phrase("406 clean evaluation prompts (every 406 in the paper)", len(keep) == 406, r"(?<![\d.])406(?![\d])", text, setting=True)
    phrase("checkpoint labels: step 0 (every step 0 in the paper)", os.path.isdir(os.path.join(_ROOT, "followup/acts/phase0_harvest_runA/0")), r"(?<![\w.])steps?[- ]0(?!\d|\.\d)", text, setting=True)
    phrase("95% intervals (every 95% in the paper)", "[2.5, 97.5]" in open(os.path.join(_ROOT, "scripts/verify_tables.py")).read()
           and "ci: float = 0.95" in open(os.path.join(_ROOT, "extension/probe/structural_baselines.py")).read(), r"(?<![\d.])95\\%(?= CI| prompt-clustered| intervals)", text, setting=True)
    phrase("layer 16 (section 3)", argparse_defaults("extension/probe/structural_baselines.py")["layer"] == 16, r"probe on layer-16\s+states reaches", text)
    phrase("reward probe on temperature-1 answers (section 3)", t1, r"separately, on temperature-1 answers to 300 prompts", text)
    phrase("arm R's probe on temperature-0.6 answers", argparse_defaults("extension/evaluation/sample_local_jsonl.py")["temperature"] == 0.6, r"trains against a probe fit on temperature-0\.6 answers", text)
    phrase("500 procedurally generated problems", len(prompts500) == 500, r"Eval is 500 procedurally generated problems filtered to the 406", text)
    phrase("3 or 4 integers per problem", {len(p["nums"]) for p in prompts500} == {3, 4}, r"Given 3--4 integers and a target", text)
    four = np.mean([len(prompts500[i]["nums"]) == 4 for i in keep])
    check("share of the 406 with four numbers (%)", 100 * four, "61", 0.5, text, r"so {n}\\% of the 406 use four numbers")
    nfeat = len(srp.load_any(os.path.join(ARMS, "probe_surface_only.pkl")).keys)
    phrase("39 surface features", nfeat == 39, r"(?<![\d.])39(?=(?: surface| text)? features|-feature)", text, setting=True)
    phrase("24 marker words", len(srp.MARKER_WORDS) == 24, r"counts of 24 marker words", text)
    sd = argparse_defaults("extension/evaluation/sample_local_jsonl.py")
    L2 = "penalty=" not in open(os.path.join(_ROOT, "extension/probe/structural_baselines.py")).read() + open(os.path.join(_ROOT, "extension/probe/save_probe_pickle_temp1.py")).read() \
        and (LogisticRegression().penalty == "l2"   # sklearn 1.8 deprecates penalty; l1_ratio=0 is L2
             or (LogisticRegression().penalty == "deprecated" and LogisticRegression().l1_ratio == 0))
    phrase("probes are L2-regularised logistic regression", L2, r"Probes are L2-regularised logistic regression", text)
    hand = open(os.path.join(ARMS, "HANDOFF.md")).read()
    yml = open(os.path.join(_ROOT, "followup", "configs", "fragility", "phase0_harvest_runA.yaml")).read()
    jd = argparse_defaults("followup/experiments/fragility/judge_lag/score_ladder_with_judge.py")
    jsrc = open(os.path.join(_ROOT, "followup/experiments/fragility/judge_lag/score_ladder_with_judge.py")).read()
    arm_cmd = re.search(r"eval_local --.*?--num_responses (\d+) --temperature ([\d.]+) --top_p ([\d.]+) --top_k (-?\d+) --max_tokens (\d+)", hand, re.S).groups()
    print(f"  (headline sampler defaults {sd.get('num_responses')}, T={sd.get('temperature')}, top-p {sd.get('top_p')}, top-k {sd.get('top_k')}; "
          f"arm command {arm_cmd}; judge defaults {jd.get('n_rollouts')}, T={jd.get('temperature')})")
    phrase("at most 1,024 new tokens everywhere", sd["max_tokens"] == 1024 and arm_cmd[4] == "1024" and "max_tokens: 1024" in yml
           and jd["max_tokens"] == 1024 and all(snap[k]["max_tokens"] == 1024 for k in snap),
           r"all on the same 406 prompts, at most 1\{,\}024 new tokens", text)
    rows = table_rows(text, r"name & answers per prompt & sampling", "name")
    compare_row("Table 1(b), headline protocol", rows, "headline",
                [str(sd["num_responses"]), str(sd["temperature"]), str(sd["top_p"]), str(sd["top_k"])])
    compare_row("Table 1(b), arm protocol", rows, "arm",
                [arm_cmd[0], "1" if float(arm_cmd[1]) == 1 else arm_cmd[1], "1" if float(arm_cmd[2]) == 1 else arm_cmd[2]])
    ladder = ("rollouts_per_prompt: 8" in yml and "temperature: 1.0" in yml and "top_p: 1.0" in yml)
    compare_row("Table 1(b), probe ladder", rows, "probe ladder", ["8", "1", "1"] if ladder else ["?"])
    judge_ok = jd["n_rollouts"] == 8 and jd["temperature"] == 1.0 and "top_p=1.0" in jsrc
    compare_row("Table 1(b), judge ladder", rows, "judge ladder", ["8", "1", "1"] if judge_ok else ["?"])
    phrase("sampling: temperature-1 for the reward probe, 0.6 for the trace-final",
           sd["temperature"] == 0.6 and all(snap[k]["temperature"] == 1 for k in snap),
           r"To match the temperature-1 answers the policy samples during RL, rather than the trace-final probe's temperature-0\.6 answers", text)
    phrase("reward probe fit on temperature-1 answers (abstract)", t1 and all(snap[k]["temperature"] == 1 for k in snap),
           r"temperature-1 answers like those reinforcement learning samples", text)
    phrase("contribution 1: second fit on temperature-1 answers", t1, r"a second fit, on temperature-1 answers; its", text)
    phrase("ladder settings, section 4.1", ladder, r"we sample 8 fresh answers per prompt at temperature 1 from that checkpoint", text)
    phrase("ladder settings, Figure 1 caption", ladder, r"8 fresh answers per prompt on the 406 prompts, sampled at temperature 1\.", text)
    phrase("ladder settings, Table 3 caption", ladder, r"probe-ladder protocol: 8 fresh answers per prompt at temperature 1 on the 406 prompts", text)
    phrase("judge ladder settings, Table 5 caption", judge_ok, r"an independent draw of 8 answers per prompt at temperature 1, graded", text)
    phrase("judge ladder settings, section 4.2", judge_ok, r"406 prompts \\times 8 answers at temperature 1, 35\{,\}728 answers", text)
    phrase("judge reads the top 20 logprobs", "logprobs=20" in jsrc, r"first generated token's top-20 logprobs", text)
    jrows = [json.loads(l) for st in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
             for l in open(os.path.join(_ROOT, f"followup/results/fragility/judge_lag/step_{st}.jsonl"))]
    phrase("two answers with no YES in the top 20 score 0", sum(r["judge_score"] == 0.0 for r in jrows) == 2 and "logprobs=20" in jsrc,
           r"Two of the 35\{,\}728 answers had no \\textsc\{yes\} variant in the top 20 and score 0\.", text)
    c, rA, aA = snap["C_outcome"], snap["runA"], snap["arm A"]
    phrase("Appendix C: C_outcome's recipe", c["num_training_steps"] == 100 and c["batch_size"] == 128 and c["group_size"] == 16
           and c["kl_divergence_coefficient"] == 1e-3 and c["entropy_coefficient"] == 1e-3 and c["learning_rate"] == 1e-5
           and c["weight_decay"] == 1e-4 and c["gradient_clipping"] == 0,
           r"is 100 RLOO\s+steps with the verifier reward at batch 128, group size 16, KL 10\^\{-3\}, entropy 10\^\{-3\}, lr 10\^\{-5\}, weight decay 10\^\{-4\} and no gradient clipping\.", text)
    phrase("Appendix C: runA and runB", all(snap[k]["group_size"] == 8 and snap[k]["weight_decay"] == 0.01 and snap[k]["gradient_clipping"] == 1 for k in ("runA", "runB")),
           r"use group size 8, weight decay 0\.01 and gradient clipping at 1\.0, with the KL anchor", text)
    phrase("Appendix C: arms A, B and R", all(snap[k]["group_size"] == 8 and snap[k]["weight_decay"] == 1e-4 and snap[k]["gradient_clipping"] == 0 for k in ("arm A", "arm B", "arm R")),
           r"arms A, B and R use group size 8, weight decay 10\^\{-4\} and no clipping", text)
    same = all(rA[k] == c[k] for k in ("learning_rate", "kl_divergence_coefficient", "entropy_coefficient", "temperature", "top_p"))
    phrase("section 4: probe-reward runs vs C_outcome's run", same and rA["num_training_steps"] == 100 and rA["group_size"] == 8 and c["group_size"] == 16,
           r"Training runs 100 RLOO steps\. Against the run that produced C_\{\\text\{outcome\}\} it keeps the learning rate, KL and entropy coefficients and sampling, but uses group size 8 rather than 16, weight decay 0\.01 rather than 10\^\{-4\}, and gradient clipping at 1\.0 rather than none", text)
    phrase("section 4.3: arms vs runB", snap["runB"]["weight_decay"] == 0.01 and aA["weight_decay"] == 1e-4,
           r"it used weight decay 0\.01 and gradient clipping at 1\.0 where the arms used 10\^\{-4\} and none", text)
    phrase("C_outcome: 100 RLOO steps (section 2)", c["num_training_steps"] == 100, r"C_\{\\text\{outcome\}\} is 100 RLOO steps against the verifier", text)
    phrase("arms: 100 RLOO steps (Figure 2, Table 2)", all(snap[k]["num_training_steps"] == 100 for k in ("arm A", "arm B", "arm R")),
           r"after 100 RLOO steps against it", text)
    phrase("arms: accuracy at step 100 (Table 2, section 4.3)", aA["num_training_steps"] == 100, r"first-block after 100 updates, all four", text)
    phrase("pre-registration: accuracy at step 100", aA["num_training_steps"] == 100, r"first-block accuracy at step 100 with the published", text)
    rpm = [snap[k]["final_logged_train_probe_mean"] for k in ("runA", "runB")]
    phrase("both runs end training at a probe score of about 0.99", all(abs(x - 0.99) < 0.005 for x in rpm),
           r"Both runs ended training with a mean probe score of about 0\.99\.", text)
    sbd = argparse_defaults("extension/probe/structural_baselines.py")
    sb_src = open(os.path.join(_ROOT, "extension/probe/structural_baselines.py")).read()
    fit_src = open(os.path.join(_ROOT, "extension/probe/save_probe_pickle_temp1.py")).read()
    import inspect
    from evaluation.countdown import compute_score
    cs_def = inspect.signature(compute_score).parameters
    yml16 = "layer: 16" in open(os.path.join(_ROOT, "followup/configs/fragility/phase0_harvest_runA.yaml")).read()
    phrase("verifier rewards {0, 0.1, 1.0}", cs_def["format_score"].default == 0.1 and cs_def["score"].default == 1.0, r"The verifier returns \\\{0, 0\.1, 1\.0\\\}", text)
    phrase("layer 16 (Appendix C)", sbd["layer"] == 16 and yml16, r"logistic regression at layer 16\.", text)
    phrase("layer 16 (Appendix D)", sbd["layer"] == 16 and yml16, r"Layer 16 was fixed before any experiment reported here", text)
    phrase("5-fold GroupKFold", "GroupKFold(5)" in sb_src and "GroupKFold(5)" in fit_src, r"scored by 5-fold", text)
    phrase("checkpoints start at step 0", os.path.isdir(os.path.join(_ROOT, "followup/acts/phase0_harvest_runA/0")), r"distribution at step 0\.", text)

    # ---------------------------------------------------------------- headline accuracies
    print("=== headline accuracies (sections 2 and 4) ===")
    acc = {k: headline(f, keep) for k, f in (("C_SFT", "eval_c_sft_n500.json"), ("C_outcome", "eval_c_outcome_n500.json"),
                                               ("runA", "eval_runA_postRL_n500.json"), ("runB", "eval_runB_postRL_n500.json"))}
    check("C_SFT, headline (section 2)", acc["C_SFT"], "0.290", 0.0005, text, r"prompts from {n} to 0\.550 \(headline")
    check("C_outcome, headline (section 2)", acc["C_outcome"], "0.550", 0.0005, text, r"from 0\.290 to {n} \(headline")
    check("runA, headline", acc["runA"], "0.236", 0.0005, text, r"falls to {n}\. That is")
    check("runA - C_outcome, pp", 100 * (acc["runA"] - acc["C_outcome"]), "-31", 0.5, text, r"That is {n}pp from")
    check("C_outcome, headline (section 4)", acc["C_outcome"], "0.550", 0.0005, text, r"from C_\{\\text\{outcome\}\}'s {n}\.")
    check("runB, headline", acc["runB"], "0.073", 0.0005, text, r"falls to {n}, which is")
    check("runB - C_SFT, pp", 100 * (acc["runB"] - acc["C_SFT"]), "-22", 0.5, text, r"which is {n}pp from")
    check("C_SFT, headline (section 4)", acc["C_SFT"], "0.290", 0.0005, text, r"from C_\{\\text\{SFT\}\}'s {n}\. Both")
    check("abstract: runA lowers accuracy by (pp)", 100 * (acc["C_outcome"] - acc["runA"]), "31", 0.5, text, r"lowers true accuracy by {n} percentage points")
    check("contribution 1: costs (pp)", 100 * (acc["C_outcome"] - acc["runA"]), "31", 0.5, text, r"it costs {n}pp when made the reward")

    # ---------------------------------------------------------------- section 3 and Appendix D
    print("=== the trace-final probe and its baselines (section 3, Appendix D) ===")
    for ctx in (r"one {n}\\% of the time", ):
        check("abstract: ranks correct above incorrect (%)", 100 * so["auroc_probe"], "98", 0.5, text, ctx)
    for lbl, ctx in (("contribution 1", r"probe, reaches AUROC {n}\. The probe trained"), ("section 3", r"states reaches AUROC {n} on C_"),
                     ("section 4.3", r"close to the probe's {n}, so much"), ("Appendix C", r"in-sample against {n} held out"),
                     ("Appendix D", r"not the raw {n}, is the number")):
        check(f"trace-final AUROC ({lbl})", so["auroc_probe"], "0.982", 0.0005, text, ctx)
    check("in-sample AUROC", so["auroc_probe_INSAMPLE"], "0.995", 0.0005, text, r"probe reads {n} in-sample")
    check("position alone", so["auroc_thinkclose_position"], "0.915", 0.0005, text, r"position alone scores {n},")
    for lbl, ctx in (("section 3", r"surface model scores {n}\."), ("section 4.3", r"activations reaches {n}, close to"),
                     ("Table 1 caption", r"surface model reads {n} under the")):
        check(f"39-feature surface model ({lbl})", so["auroc_surface39"], "0.926", 0.0005, text, ctx)
    sh = so["probe_share_of_headroom"]
    for lbl, a_, b_, c_ in (("section 3", r"captures {n}\\% of the accuracy gain", r"gain available to any selector \(CI \[{n}\\%", r"selector \(CI \[78\.9\\%, {n}\\%\]\)"),
                            ("Appendix D", r"that is {n}\\% of the available gain", r"bootstrap, CI \[{n}\\%, 96", r"bootstrap, CI \[78\.9\\%, {n}\\%\]\)")):
        check(f"share of headroom ({lbl})", 100 * sh["est"], "88.8", 0.05, text, a_)
        check(f"  its CI low ({lbl})", 100 * sh["ci_lo"], "78.9", 0.05, text, b_)
        check(f"  its CI high ({lbl})", 100 * sh["ci_hi"], "96.5", 0.05, text, c_)
    cs, cf = so["contrasts"]["probe_minus_shortest"], so["contrasts"]["probe_minus_surface39"]
    check("probe over free baselines, pp (section 3)", 100 * cs["est"], "+3.20", 0.005, text, r"free baselines by {n}pp of selection")
    check("probe over free baselines, pp (Appendix D)", 100 * cs["est"], "+3.20", 0.005, text, r"free baselines is {n}pp of selection")
    phrase("both free baselines pick the same answers", abs(cs["est"] - cf["est"]) < 1e-9, r"beats both free baselines", text)
    check("  its CI low", 100 * cs["ci_lo"], "+0.99", 0.005, text, r"accuracy \(CI \[{n}, \+5\.42\]")
    check("  its CI high", 100 * cs["ci_hi"], "+5.42", 0.005, text, r"\[\+0\.99, {n}\], p")
    check("  p against shortest <think>", cs["p"], "0.008", 0.0005, text, r"p = {n}\s+against picking the answer with the shortest")
    check("  p against the surface model", cf["p"], "0.004", 0.0005, text, r"and p = {n} against the 39-feature")
    check("AUROC margin over the stronger baseline", so["auroc_probe"] - max(so["auroc_surface39"], so["auroc_thinkclose_position"]), "+0.056", 0.0005, text, r"and {n} AUROC over the stronger")
    check("selector: probe", so["select_probe"], "0.6700", 0.00005, text, r"the probe reaches {n}, against")
    check("selector: random", so["select_random"], "0.5531", 0.00005, text, r"against {n} for picking at random")
    check("selector: oracle", so["select_oracle"], "0.6847", 0.00005, text, r"and {n} for an oracle")
    phrase("best-of-16 selector, up to 16 scoreable answers", so.get("select_min_effective_K", 0) >= 1 and sb["C_outcome"]["n_prompts"] == 406,
           r"As a best-of-16 selector on 406 held-out prompts", text)
    phrase("best-of-16 (section 3)", sbd["K"] == 16, r"As a best-of-16 selector it captures", text)
    phrase("best-of-16 headroom (Appendix D)", sbd["K"] == 16, r"of the best-of-16 selection headroom", text)
    check("mean scoreable answers per prompt", so["select_mean_effective_K"], "15.5", 0.05, text, r"up to 16 scoreable answers, {n} on average")
    phrase("up to 16 answers per prompt", sbd["K"] == 16 and so["select_frac_prompts_full_K"] > 0, r"A prompt has up to 16 scoreable answers", text)
    check("probe, stratified by position", so["auroc_probe_stratified_by_position"], "0.939", 0.0005, text, r"probe still reads {n}, against")
    check("position, stratified the same way", so["auroc_position_stratified_by_position"], "0.564", 0.0005, text, r"against {n} for position alone")
    check("shortest <think> share of selection headroom (%)",
          100 * (so["select_shortest_think"] - so["select_random"]) / (so["select_oracle"] - so["select_random"]), "64", 0.5, text,
          r"already captures {n}\\% of the best-of-16")
    for nm, r_, w in (("C_outcome", so, "2.9"), ("C_SFT", ss, "10.4")):
        check(f"unscoreable share, {nm} (%)", 100 * r_["n_without_think_close"] / (r_["n_with_think_close"] + r_["n_without_think_close"]), w, 0.05, text,
              r"\({n}\\% on C_" if nm == "C_outcome" else r", {n}\\% on C_\{\\text\{SFT\}\}, all")
    phrase("unscoreable answers are all wrong", so["acc_without_think_close"] == 0 and ss["acc_without_think_close"] == 0, r"10\.4\\% on C_\{\\text\{SFT\}\}, all of them wrong", text)
    # the reward probe's own protocol: refit exactly as save_probe_pickle_temp1.py did
    P = os.path.join(_ROOT, "extension", "cache", "probe_cache_temp1", "C_outcome_temp1_l16_pre_answer")
    X = np.load(P + ".npz")["X"]
    meta = json.load(open(P + ".meta.json"))
    fit_rows = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_c_outcome_temp1_asingh_300.json")) if l.strip()]
    labs = {(p_, i): int(sc == 1.0) for p_, row in enumerate(fit_rows) for i, sc in enumerate(row["scores"])}
    y = np.array([labs.get((int(m["prompt_idx"]), int(m["resp_idx"])), 0) for m in meta])
    grp = np.array([int(m["prompt_idx"]) for m in meta])
    sc = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=2000))]).fit(X[tr], y[tr])
        sc[te] = pipe.predict_proba(X[te])[:, 1]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    nb = min(len(pos), len(neg))
    rs = np.random.RandomState(0)
    idx = np.concatenate([rs.choice(pos, nb, replace=False), rs.choice(neg, nb, replace=False)])
    a810 = roc_auc_score(y[idx], sc[idx])
    check("reward probe, its own protocol (section 3)", a810, "0.810", 0.0005, text, r"on that protocol is {n}\.")
    check("reward probe, its own protocol (Table 1 caption)", a810, "0.810", 0.0005, text, r"reads {n} on its own protocol")
    check("reward probe's fitting prompts", len(fit_rows), "300", 0, text, r"answers to {n} prompts from the RL training pool")
    phrase("none of them among the 406", vrl.fitting_overlap() == 0, r"none of them among the 406\)", text)
    X4 = np.load(vrl.POP + ".npz", allow_pickle=True)["X"]
    m4 = json.load(open(vrl.POP + ".meta.json"))
    import pickle
    rp = pickle.load(open(vrl.REWARD, "rb"))
    check("reward probe, all 406, first block", roc_auc_score(vrl._first_block_labels(m4), rp.predict_proba(X4)[:, 1]), "0.951", 0.0005, text, r"block it reads {n} over all 406")
    tf = pickle.load(open(os.path.join(_ROOT, "extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl"), "rb"))
    w_in = lambda pipe: pipe.steps[-1][1].coef_.ravel() / pipe.steps[0][1].scale_
    wr, wt = w_in(rp), w_in(tf)
    cos = wr @ wt / np.linalg.norm(wr) / np.linalg.norm(wt)
    check("cosine, reward vs trace-final (section 3)", cos, "0.192", 0.0005, text, r"little direction \(cosine {n}\)")
    check("cosine, reward vs trace-final (contribution 1)", cos, "0.192", 0.0005, text, r"direction has cosine {n} with the first's")
    raw = srp.load_any(os.path.join(ARMS, "probe_raw_arm_recipe.pkl"))
    wR = raw.pipe.w / raw.pipe.sd
    check("cosine, arm R probe vs trace-final", wR @ wt / np.linalg.norm(wR) / np.linalg.norm(wt), "0.779", 0.0005, text, r"at cosine {n} with the trace-final probe")

    # ---------------------------------------------------------------- the template score
    print("=== the template score (section 4) ===")
    for f, w_m, w_r, cm, cr in (("eval_c_outcome_n500.json", "1.81", "+0.643", r"rises {n} \\to 3\.23 from", r"falls from r={n} to \+0\.068"),
                                ("eval_runA_postRL_n500.json", "3.23", "+0.068", r"1\.81 \\to {n} from C_", r"r=\+0\.643 to {n} and from"),
                                ("eval_c_sft_n500.json", "3.24", "+0.168", r"and {n} \\to 4\.32 from C_", r"and from {n} to \+0\.023"),
                                ("eval_runB_postRL_n500.json", "4.32", "+0.023", r"3\.24 \\to {n} from C_", r"from \+0\.168 to {n}\. The probe")):
        rows_ = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
        ts = np.array([template_score(x) for i, r_ in enumerate(rows_) if i in keep for x in r_["response"]])
        yy = np.array([first_block(x, r_["nums"], r_["target"]) for i, r_ in enumerate(rows_) if i in keep for x in r_["response"]])
        check(f"template score, {f}", ts.mean(), w_m, 0.005, text, cm)
        check(f"template r with correctness, {f}", np.corrcoef(ts, yy)[0, 1], w_r, 0.0005, text, cr)
    from quantify_structural_confound import TEMPLATE_PATTERNS
    phrase("seven template phrases", len(TEMPLATE_PATTERNS) == 7, r"counting which of seven reasoning phrases appear", text)

    # ---------------------------------------------------------------- the arms (section 4.3, Table 2)
    print("=== the reward arms (section 4.3, Table 2, Appendix C) ===")
    cut_o = vrl.arm_aurocs("C_outcome", cut=True)
    ev_o = vrl.arm_aurocs("C_outcome", cut=False)
    cut_s = vrl.arm_aurocs("C_SFT", cut=True)
    kp = arms_gate.clean_prompts()
    per = {k: arms_gate.load_arm(f, kp) for k, f in (("A", "eval_armA_residual_step100.json"), ("B", "eval_armB_surface_step100.json"),
                                                     ("R", "eval_armRaw_step100.json"), ("C", "eval_runB_armprotocol_step100.json"),
                                                     ("Chead", "eval_runB_postRL_n500.json"))}
    aacc = {k: float(np.mean(np.concatenate(list(v.values())))) for k, v in per.items()}
    boot = {k: arms_gate.paired_bootstrap(per[k], per["A"], 10000, 0) for k in "BRC"}
    rows = table_rows(text, r"Arm & reward & training cut & eval.\ text", "Arm")
    f4 = lambda x: f"{x:.4f}"
    f2 = lambda x: f"{x:.2f}"
    compare_row("Table 2, arm A", rows, "A", [f"{cut_o['A']:.3f}", f"{ev_o['A']:.3f}", f4(aacc["A"])])
    for k, key in (("B", "B"), ("R", "R"), ("C", "C^{\\dagger}")):
        d, lo, hi = boot[k]
        compare_row(f"Table 2, arm {k}", rows, key, [f"{cut_o[k]:.3f}", f"{ev_o[k]:.3f}", f4(aacc[k]), f2(d), f2(lo), f2(hi)])
    for lbl, w, ctx in (("A", "0.958", r"A {n}, B 0\.900 and R"), ("B", "0.900", r"A 0\.958, B {n} and R"), ("R", "0.978", r"B 0\.900 and R {n};")):
        check(f"arm {lbl} AUROC, as trained (section 4.3)", cut_o[lbl], w, 0.0005, text, ctx)
    for lbl, w, ctx in (("A", "0.834", r"uncut, A {n}, B 0\.925"), ("B", "0.925", r"A 0\.834, B {n} and R"), ("R", "0.978", r"B 0\.925 and R {n}\.")):
        check(f"arm {lbl} AUROC, evaluation text", ev_o[lbl], w, 0.0005, text, ctx)
    for lbl, w, ctx in (("A", "0.721", r"lower: A {n}, B"), ("B", "0.660", r"A 0\.721, B {n} and"), ("R", "0.816", r"B 0\.660 and R {n}\.")):
        check(f"arm {lbl} AUROC through C_SFT, as trained", cut_s[lbl], w, 0.0005, text, ctx)
    import fast_boot as FB
    for model, w_pt, w_lo, w_hi, ctx_pt in (("C_outcome", "0.020", "0.012", "0.028", r"every reading, by {n} \(CI"), ("C_SFT", "0.095", "0.078", "0.112", r"and by {n} \(CI \[0\.078")):
        Xh, Th, yh, ph = vrl.heldout_population(model, with_prompts=True)
        Thc = [vrl.cut_at_answer(t) for t in Th]
        pr = np.unique(ph)
        Cb = FB.draws(pr, 2000, 0)
        ckA = FB.Checkpoint(srp.load_any(os.path.join(ARMS, "probe_surface_residual_l16.pkl")).predict_proba(Xh, text=Thc)[:, 1], yh, ph, pr)
        ckR = FB.Checkpoint(srp.load_any(os.path.join(ARMS, "probe_raw_arm_recipe.pkl")).predict_proba(Xh, text=Thc)[:, 1], yh, ph, pr)
        dd = ckR.auroc(Cb) - ckA.auroc(Cb)
        lo_, hi_ = np.percentile(dd, [2.5, 97.5])
        gap = (cut_o if model == "C_outcome" else cut_s)
        check(f"R - A AUROC gap, {model}", gap["R"] - gap["A"], w_pt, 0.0005, text, ctx_pt)
        check(f"  its CI low, {model}", lo_, w_lo, 0.0005, text, r"\(CI \[{n}, " + re.escape(w_hi) + r"\]\)")
        check(f"  its CI high, {model}", hi_, w_hi, 0.0005, text, r"\(CI \[" + re.escape(w_lo) + r", {n}\]\)")
    phrase("arm B reads lowest on answers cut as in training", cut_o["B"] < min(cut_o["A"], cut_o["R"], cut_o["C"]), r"read lowest \(0\.900\)", text)
    check("arm B AUROC, cut (abstract, contribution 1)", cut_o["B"], "0.900", 0.0005, text, r"read lowest \({n}\)")
    check("arm B, Table 1 (shipped artifact)", ev_o["B"], "0.925", 0.0005, text, r"and {n} as the shipped arm-B")
    check("reward probe on the arms' population (Table 1 caption)", cut_o["C"], "0.945", 0.0005, text, r"and {n} on the population the reward arms share")
    check("reward probe on the arms' population (section 3)", cut_o["C"], "0.945", 0.0005, text, r"\({n} on the half used in")
    rows1 = table_rows(text, r"name & what it reads & what it was fit to & AUROC", "name")
    compare_row("Table 1(a), trace-final probe", rows1, "trace-final probe", [str(sbd["layer"]), f"{so['auroc_probe']:.3f}"])
    compare_row("Table 1(a), reward probe", rows1, "reward probe", [str(sbd["layer"]), "1" if t1 else "?", f"{a810:.3f}", f"{cut_o['C']:.3f}"])
    compare_row("Table 1(a), surface model", rows1, "surface model", [str(nfeat), f"{so['auroc_surface39']:.3f}", f"{ev_o['B']:.3f}"])
    j0 = [json.loads(l) for l in open(os.path.join(_ROOT, "followup/results/fragility/judge_lag/step_0.jsonl"))]
    compare_row("Table 1(a), LLM judge", rows1, "LLM judge", [f"{roc_auc_score([r_['correct'] for r_ in j0], [r_['judge_score'] for r_ in j0]):.3f}"])
    for lbl, w, ctx in (("abstract", "0.958", r"regressed out \(AUROC {n}\)"), ("contribution 1", "0.958", r"\(arm A\) reads {n} and left")):
        check(f"arm A as trained ({lbl})", cut_o["A"], w, 0.0005, text, ctx)
    for lbl, ctx in (("abstract", r"read higher \({n}\) and left"), ("contribution 1", r"\(arm R\) reads {n} and left")):
        check(f"arm R ({lbl})", cut_o["R"], "0.978", 0.0005, text, ctx)
    check("arm A accuracy (abstract)", aacc["A"], "0.168", 0.0005, text, r"left the most accuracy \({n}\)")
    check("arm A accuracy (contribution 1)", aacc["A"], "0.168", 0.0005, text, r"reads 0\.958 and left {n} accuracy")
    check("arm R accuracy (abstract)", aacc["R"], "0.051", 0.0005, text, r"\(0\.978\) and left {n}\.")
    check("arm R accuracy (contribution 1)", aacc["R"], "0.051", 0.0005, text, r"reads 0\.978 and left {n}, ")
    check("arm B accuracy (contribution 1)", aacc["B"], "0.0000", 0.00005, text, r"accuracy to exactly {n}\. Arm A's")
    dR, loR, hiR = arms_gate.paired_bootstrap(per["A"], per["R"], 10000, 0)
    check("arm A - arm R, pp (contribution 1)", dR, "11.6", 0.05, text, r"0\.051, {n}pp less")
    check("  its CI low", loR, "9.6", 0.05, text, r"less \(CI \[{n}, 13\.8\]\)")
    check("  its CI high", hiR, "13.8", 0.05, text, r"\(CI \[9\.6, {n}\]\)\. A reward")
    check("arm A - arm R, pp (section 4.3)", dR, "+11.64", 0.005, text, r"kept {n}pp more accuracy")
    check("  its CI low", loR, "+9.61", 0.005, text, r"accuracy \(CI \[{n}, \+13\.76\]\)")
    check("  its CI high", hiR, "+13.76", 0.005, text, r"\(CI \[\+9\.61, {n}\]\)")
    check("run ended lower, pp (section 4.3)", dR, "11.6", 0.05, text, r"its run ended {n}pp lower")
    dBR = arms_gate.paired_bootstrap(per["B"], per["R"], 10000, 0)
    check("arm B - arm R, pp", dBR[0], "-5.14", 0.005, text, r"either way \({n}pp, CI")
    check("  its CI low", dBR[1], "-6.25", 0.005, text, r"CI \[{n}, -4\.09\]\)")
    check("  its CI high", dBR[2], "-4.09", 0.005, text, r"\[-6\.25, {n}\]\), and against")
    check("arm A - runA (cross-protocol), pp", 100 * (acc["runA"] - aacc["A"]), "6.83", 0.005, text, r"it ended {n}pp below")
    check("arm A's fall from C_SFT, pp", 100 * (acc["C_SFT"] - aacc["A"]), "12.3", 0.05, text, r"arm A fell {n}pp from")
    check("runA's fall from C_outcome, pp", 100 * (acc["C_outcome"] - acc["runA"]), "31.4", 0.05, text, r"{n}pp from C_\{\\text\{outcome\}\}\. Arm B's")
    phrase("arm B vs runA holds only as final accuracy", aacc["B"] < acc["runA"] and (acc["C_SFT"] - aacc["B"]) < (acc["C_outcome"] - acc["runA"])
           and (acc["C_SFT"] - aacc["B"]) > (acc["C_SFT"] - aacc["R"]), r"and against\s+only as final accuracy", text)
    check("residualisation's AUROC cost, as trained", cut_o["R"] - cut_o["A"], "0.020", 0.0005, text, r"AUROC by {n} on answers cut as in training")
    check("residualisation's AUROC cost, evaluation text", ev_o["R"] - ev_o["A"], "0.144", 0.0005, text, r"or by {n} uncut")
    pp = {k: arms_gate.paired_bootstrap(per["C"], per["Chead"], 10000, 0) for k in ("x",)}["x"]
    check("arm C, headline protocol", aacc["Chead"], "0.0734", 0.00005, text, r"headline protocol, at {n};")
    check("arm C, arm protocol", aacc["C"], "0.0782", 0.00005, text, r"three it reads {n}, a difference")
    check("protocol difference, pp", pp[0], "+0.48", 0.005, text, r"a difference of {n}pp")
    check("  its CI low", pp[1], "-0.62", 0.005, text, r"\(CI \[{n}, \+1\.60\]\)")
    check("  its CI high", pp[2], "+1.60", 0.005, text, r"\[-0\.62, {n}\]\), so arm C's")
    # arm A's reward, decomposed on the text it scored
    A = srp.load_any(os.path.join(ARMS, "probe_surface_residual_l16.pkl"))
    B = srp.load_any(os.path.join(ARMS, "probe_surface_only.pkl"))
    Xo, To, yo = vrl.heldout_population("C_outcome")
    Toc = [vrl.cut_at_answer(t) for t in To]
    k = len(A.keys)
    tcoef = -(A.B[:k] @ (A.pipe.w / A.pipe.sd))
    act = ((Xo - A.pipe.mu) / A.pipe.sd) @ A.pipe.w
    txt = ((srp.feature_matrix(Toc, A.keys) - A.mu) / A.sigma) @ tcoef
    if abs(roc_auc_score(yo, act + txt) - cut_o["A"]) > 1e-6:
        bad.append("arm A's decomposition does not reproduce its own AUROC")
    check("activation term alone", roc_auc_score(yo, act), "0.976", 0.0005, text, r"activation term alone reads {n}")
    check("text term alone", roc_auc_score(yo, txt), "0.094", 0.0005, text, r"text term alone {n}, and")
    check("the two together", cut_o["A"], "0.958", 0.0005, text, r"the two together {n}\.")
    bw = np.asarray(B.scorer.w).ravel() / np.asarray(B.scorer.sd).ravel() * A.sigma
    varies = (np.abs(tcoef) > 1e-12) | (np.abs(bw) > 1e-12)
    check("features that vary", int(varies.sum()), "27", 0, text, r"on 24 of the {n} features that vary")
    check("of those, opposite in sign to arm B's", int((np.sign(tcoef[varies]) != np.sign(bw[varies])).sum()), "24", 0, text, r"on {n} of the 27 features")
    check("correlation with arm B's weights", np.corrcoef(tcoef, bw)[0, 1], "-0.56", 0.005, text, r"vary \(r={n}\)")
    def text_term(f):
        rows_ = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
        tx = [vrl.cut_at_answer(x) for i, r_ in enumerate(rows_) if i in keep for x in r_["response"]]
        return float((((srp.feature_matrix(tx, A.keys) - A.mu) / A.sigma) @ tcoef).mean()), tx
    tb, _ = text_term("eval_armB_surface_step100.json")
    tsft, tx_sft = text_term("eval_c_sft_n500.json")
    check("text term on arm B's final answers (logits)", tb, "-24.2", 0.05, text, r"contributes {n} logits")
    check("text term on C_SFT's answers (logits)", tsft, "+2.82", 0.005, text, r"against {n} on C_")
    tl = np.array([srp.surface_features(t)["tail_len"] for t in To])
    tlc = np.array([srp.surface_features(t)["tail_len"] for t in Toc])
    check("text after </think>, evaluation answers (chars)", tl.mean(), "1{,}160", 0.5, text, r"about {n} characters on evaluation")
    check("text after </think>, as trained (chars)", tlc.mean(), "61", 0.5, text, r"and {n} on the same answers cut")
    check("its correlation with correctness, evaluation", np.corrcoef(tl, yo)[0, 1], "+0.72", 0.005, text, r"changes sign \({n} to")
    check("its correlation with correctness, as trained", np.corrcoef(tlc, yo)[0, 1], "-0.20", 0.005, text, r"\(\+0\.72 to {n}\)")
    rows_sft = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_c_sft_n500.json")) if l.strip()]
    tx_full = [x for i, r_ in enumerate(rows_sft) if i in keep for x in r_["response"]]
    zeros = lambda n: np.zeros((n, 1))
    check("arm B's reward on C_SFT's answers, as trained", B.predict_proba(zeros(len(tx_sft)), text=tx_sft)[:, 1].mean(), "0.12", 0.005, text, r"ordinary answers low: {n} on average")
    check("arm B's reward on the same answers uncut", B.predict_proba(zeros(len(tx_full)), text=tx_full)[:, 1].mean(), "0.53", 0.005, text, r"against {n} on the same answers uncut")
    w = dict(zip(B.keys, np.asarray(B.scorer.w).ravel()))
    order = sorted(w, key=lambda q: -abs(w[q]))[:3]
    phrase("arm B's three largest weights", set(order) == {"tail_len", "ans_len", "n_equals"}, r"The reward's largest weights favour", text)
    check("arm B weight tail_len", w["tail_len"], "+1.19", 0.005, text, r", {n} in standardised units")
    check("arm B weight ans_len", w["ans_len"], "+0.62", 0.005, text, r", {n}\), and penalise")
    check("arm B weight n_equals", w["n_equals"], "-1.11", 0.005, text, r", {n}\)\. They were fit")
    rowsB = [json.loads(l) for l in open(os.path.join(_ROOT, "eval_armB_surface_step100.json")) if l.strip()]
    allb = [x for i, r_ in enumerate(rowsB) if i in keep for x in r_["response"]]
    has = [AN.search(x) for x in allb]
    has = [m for m in has if m]
    phrase("every arm-B answer emits </think>", all("</think>" in x for x in allb), r"Every answer still emits\s+and 3\{,\}244", text)
    check("arm B answers with an <answer> block", len(has), "3{,}244", 0, text, r"and {n} of 3\{,\}248 contain")
    check("arm B answers", len(allb), "3{,}248", 0, text, r"3\{,\}244 of {n} contain")
    check("arm B blocks that are arithmetic (%)", 100 * sum(evaluate_equation(m.group(1).strip()) is not None for m in has) / len(has), "0.12", 0.005, text, r"only {n}\\% of those blocks")
    # Appendix C: accepting a trailing "= target"
    def eq_target(resp, nums, tgt):
        mm = AN.search(resp)
        if not mm:
            return 0
        e = mm.group(1).strip()
        if "=" in e:
            lhs, _, rhs = e.rpartition("=")
            if rhs.strip() != str(tgt):
                return 0
            e = lhs.strip()
        if not validate_equation(e, list(nums)):
            return 0
        v = evaluate_equation(e)
        return int(v is not None and abs(v - int(tgt)) < 1e-5)
    gain = {}
    for kk, f in (("A", "eval_armA_residual_step100.json"), ("B", "eval_armB_surface_step100.json"), ("R", "eval_armRaw_step100.json"), ("C", "eval_runB_armprotocol_step100.json")):
        rows_ = [json.loads(l) for l in open(os.path.join(_ROOT, f)) if l.strip()]
        g = [eq_target(x, r_["nums"], r_["target"]) for i, r_ in enumerate(rows_) if i in keep for x in r_["response"]]
        gain[kk] = 100 * (np.mean(g) - aacc[kk])
    print(f"  (accepting '= target' would add {gain} pp)")
    check("largest gain from accepting '= target', pp", max(gain.values()), "1.1", 0.05, text, r"add at most {n}pp to any arm's accuracy \(arm R's\)")
    if max(gain, key=gain.get) != "R":
        bad.append("the largest '= target' gain is not arm R's")
    phrase("pre-registration lists 0.978, arm R's probe", "| 0.978 |" in open(os.path.join(ARMS, "PREREGISTRATION.md")).read() and abs(cut_o["R"] - 0.978) < 0.0005,
           r"lists AUROC 0\.978, the value of arm R's probe", text)

    # ---------------------------------------------------------------- Appendix C reproducibility, Appendix F
    man = [l.split("\t") for l in open(os.path.join(_ROOT, "artifacts", "MANIFEST.tsv")).read().splitlines()[1:]]
    check("cached artifacts, GB", sum(int(r_[1]) for r_ in man) / 1e9, "0.37", 0.005, text, r"the {n}GB of cached artifacts")
    print("=== Appendix F: the quoted answer ===")
    ex = next((r_, x) for i, r_ in enumerate(rowsB) for x in r_["response"] if "56 - (29 * 38 / 38) = 68" in x)
    r_, x = ex
    blk = AN.search(x).group(1).strip().split("\n")
    quote = re.search(r"<answer>\\\\\n(.*?)\n</answer>", text, re.S)
    qlines = [q.rstrip("\\").strip() for q in quote.group(1).split("\n")] if quote else []
    ok = sorted(r_["nums"]) == sorted([29, 17, 63, 38]) and int(r_["target"]) == 68 and qlines == [b.strip() for b in blk]
    phrase("Appendix F: the quote is arm B's answer, verbatim", ok, r"<answer>\\\\\nLet me verify:\\\\\n1\. 17 - 10 = 7\\\\\n2\. 63 - 7 = 56\\\\\n3\. 29 \* 38 = 1102\\\\\n4\. 56 - \(29 \* 38 / 38\) = 68\\\\\n</answer>", text)
    phrase("Appendix F: input and target", ok, r"For input \[29, 17, 63, 38\] and target 68,", text)
    phrase("Appendix F: step 1 uses 10, not in the input", 10 not in r_["nums"], r"Step 1 uses 10, which is not in the input", text)
    phrase("Appendix F: step 3 correct", 29 * 38 == 1102, r"Step 3 is correct but irrelevant", text)
    phrase("Appendix F: step 4 evaluates to 27", abs(evaluate_equation("56 - (29 * 38 / 38)") - 27) < 1e-9,
           r"Step 4 reuses two numbers and evaluates to 27 while asserting 68", text)

    import verify_tables as VT
    bad.extend(VT.bad)                 # compare_row records Table 1 and 2 mismatches there
    L.dump("verify_claims")
    if bad:
        print(f"\nFAIL ({len(bad)})")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("\nAll claims verified.")


if __name__ == "__main__":
    main()
