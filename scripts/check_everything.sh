#!/usr/bin/env bash
# Every gate this project has, in one command. Run it after any edit to the paper
# or the analysis code, and especially after a prose rewrite -- rewriting is when
# numbers get retyped, and retyping is how every defect in the August audit got in.
#
#   bash scripts/check_everything.sh
#
# Exits non-zero if anything disagrees. Each verifier prints the published value
# beside the recomputed one, so a failure tells you which number moved.
#
# Requires the two Arm A/B rollout JSONs, which are ~10 MB each and not in git:
#   modal volume get default-proj-training \
#     evaluation/eval_results/armA_residual_step100.json ./eval_armA_residual_step100.json
#   modal volume get default-proj-training \
#     evaluation/eval_results/armB_surface_step100.json ./eval_armB_surface_step100.json

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

fail=0
run() {
  local name="$1"; shift
  printf '%-46s' "$name"
  if out=$("$@" 2>&1); then
    echo "PASS"
  else
    echo "FAIL"
    echo "$out" | tail -12 | sed 's/^/      /'
    fail=1
  fi
}

echo "=== analysis: does the code still produce the published numbers? ==="
run "structural baselines (selection, length)" \
    python -W ignore extension/probe/structural_baselines.py --out /tmp/_sb.txt
run "surface battery (the decomposition)" \
    python -W ignore extension/probe/surface_battery.py --out /tmp/_sbat.txt
run "template confound (section 3 table)" \
    python -W ignore scripts/quantify_structural_confound.py
run "pre-registered arms A/B + output shape" \
    python -W ignore extension/probe/verify_residual_arms.py
run "the 40-step lag + scope condition" \
    python followup/experiments/fragility/phase0_replicate/verify_lag_result.py
# Needs the cached activations under followup/acts/ (gitignored, ~56 MB/checkpoint).
if [ -d followup/acts/phase0_harvest_runA/50 ]; then
  run "change-point test (prompt-clustered)" \
      python -W ignore followup/experiments/fragility/phase0_replicate/changepoint_lag.py --n_boot 500 --out /tmp/_cp.txt
else
  printf '%-46s%s\n' "change-point test (prompt-clustered)" "SKIP (acts not cached)"
fi

if [ -d followup/results/fragility/judge_lag ]; then
  run "LLM judge lag (judge vs verifier)" \
      python -W ignore followup/experiments/fragility/judge_lag/analyze_judge_lag.py --n_boot 500 --out /tmp/_jl.txt
else
  printf '%-46s%s\n' "LLM judge lag (judge vs verifier)" "SKIP (scores not pulled)"
fi

if [ -d followup/results/fragility/judge_lag ]; then
  run "judge error decomposition" \
      python -W ignore scripts/verify_judge_errors.py --out /tmp/_je.txt
else
  printf '%-46s%s\n' "judge error decomposition" "SKIP (scores not pulled)"
fi

echo
echo "=== paper: do the tables in the tex match the analysis? ==="
# Added after an edit pass overwrote the probe's lag table with the judge's and
# every existing gate stayed green: they all checked that the analysis
# reproduced, none that the PAPER agreed with it.
# Three cuts of the same paper for the three JUDGe tracks: 6pp full, 4pp short,
# 2pp junior spotlight. Every one of them gets checked -- a number corrected in
# one cut and missed in another is exactly the drift these gates exist to catch.
for tex in writeup_judge writeup_judge_short writeup_judge_spotlight writeup_judge_shortpaper writeup_interpscience; do
  if [ -f "$tex.tex" ] && [ -d followup/acts/phase0_harvest_runA/50 ]; then
    run "lag tables: $tex" \
        python -W ignore scripts/verify_paper_tables.py --tex "$tex.tex"
  else
    printf '%-46s%s\n' "lag tables: $tex" "SKIP (tex or acts missing)"
  fi
done

# Table cells were gated long before prose numbers were, and the 2pp rewrite
# retyped ~20 figures into sentences where nothing checked them. This closes it.
for tex in writeup_judge_spotlight writeup_judge_shortpaper; do
  if [ -f "$tex.tex" ] && [ -d followup/results/fragility/judge_lag ]; then
    run "prose numbers: $tex" \
        python -W ignore scripts/verify_prose_numbers.py --tex "$tex.tex"
  else
    printf '%-46s%s\n' "prose numbers: $tex" "SKIP (tex or scores missing)"
  fi
done

echo
echo "=== submission: is the thing you are about to upload safe? ==="
for tex in writeup_judge writeup_judge_short writeup_judge_spotlight writeup_judge_shortpaper writeup_interpscience; do
  if [ -f "$tex.tex" ]; then
    run "anonymity scan: $tex" \
        python scripts/make_submission_tex.py --check "$tex.tex"
  else
    printf '%-46s%s\n' "anonymity scan: $tex" "SKIP (not built)"
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "All gates pass."
else
  echo "SOMETHING DISAGREES -- see above. Do not submit until this is green."
fi
exit "$fail"
