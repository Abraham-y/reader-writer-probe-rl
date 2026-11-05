#!/usr/bin/env bash
# Every gate this project has, in one command. Run it after any edit to the paper
# or the analysis code, and especially after a prose rewrite -- rewriting is when
# numbers get retyped, and retyping is how every defect in the August audit got in.
#
#   bash scripts/check_everything.sh
#
# Exits non-zero if anything disagrees. Each verifier prints the published value
# beside the recomputed one, so a failure tells you which number moved. Every
# check that matches a number in the paper registers the pattern it matched;
# verify_numbers.py then fails on any number no passing check claims, and
# mutation_test.py plants errors to show the suite catches them.
#
# Needs the cached artifacts: python scripts/artifacts.py fetch

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# One scratch directory per run, so two runs (or a run racing an edit) cannot
# read each other's ledgers; it is removed on exit.
WORK=$(mktemp -d "${TMPDIR:-/tmp}/check_everything.XXXXXX") || exit 1
trap 'rm -rf "$WORK"' EXIT
export LEDGER_DIR="$WORK/ledger"
export SB_JSON="$WORK/sb.json"

fail=0
skipped=0
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
    python -W ignore extension/probe/structural_baselines.py --out "$WORK/sb.txt"
run "surface battery (the decomposition)" \
    python -W ignore extension/probe/surface_battery.py --out "$WORK/sbat.txt" --fig "$WORK/dose.png"
run "template confound (section 3 table)" \
    python -W ignore scripts/quantify_structural_confound.py
run "pre-registered arms A/B + output shape" \
    python -W ignore extension/probe/verify_residual_arms.py --out "$WORK/arms.txt"
# verify_lag_result.py is no longer run here: it checks a different estimator's
# values (0.787, 0.772, ...) that the paper does not print. The lag the paper
# reports is gated by the change-point test and the table check below.
# Needs the cached activations under followup/acts/ (gitignored, ~56 MB/checkpoint).
if [ -d followup/acts/phase0_harvest_runA/50 ]; then
  run "change-point test (prompt-clustered)" \
      python -W ignore followup/experiments/fragility/phase0_replicate/changepoint_lag.py --n_boot 500 --out "$WORK/cp.txt"
else
  printf '%-46s%s\n' "change-point test (prompt-clustered)" "SKIP (acts not cached)"; skipped=$((skipped+1))
fi

if [ -d followup/results/fragility/judge_lag ]; then
  run "LLM judge lag (judge vs verifier)" \
      python -W ignore followup/experiments/fragility/judge_lag/analyze_judge_lag.py --n_boot 500 --out "$WORK/jl.txt"
else
  printf '%-46s%s\n' "LLM judge lag (judge vs verifier)" "SKIP (scores not pulled)"; skipped=$((skipped+1))
fi

if [ -d followup/results/fragility/judge_lag ]; then
  run "judge error decomposition" \
      python -W ignore scripts/verify_judge_errors.py --out "$WORK/je.txt"
else
  printf '%-46s%s\n' "judge error decomposition" "SKIP (scores not pulled)"; skipped=$((skipped+1))
fi

echo
echo "=== paper: do the tables in the tex match the analysis? ==="
# Added after an edit pass overwrote the probe's lag table with the judge's and
# every existing gate stayed green: they all checked that the analysis
# reproduced, none that the PAPER agreed with it.
# Only the camera-ready is gated; the earlier cuts (four JUDGe tracks and the
# long workshop draft) were removed from the repository before release.
# verify_tables.py rebuilds Tables 3-5 cell by cell (every interval and p-value
# too) and the prose statistics drawn from the two ladders; it replaced
# verify_paper_tables.py, which skipped 96 of the 272 cells.
if [ -f writeup_interpscience.tex ] && [ -d followup/acts/phase0_harvest_runA/50 ]; then
  run "ladder tables and statistics" \
      python -W ignore scripts/verify_tables.py --tex writeup_interpscience.tex
else
  printf '%-46s%s\n' "ladder tables and statistics" "SKIP (tex or acts missing)"; skipped=$((skipped+1))
fi

# The reward ladder's three read-only AUROCs were the paper's load-bearing
# uncheckable numbers: two ship inside their arms' artifacts, and the third was
# quoted for three commits as 0.978 -- a raw probe fit inside the arms script,
# not the reward. This recomputes arm C from the shipped reward pickle and
# fails if the inversion the paper claims stops holding.
if [ -f writeup_interpscience.tex ] && [ -f extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl ]; then
  run "reward ladder: writeup_interpscience" \
      python -W ignore scripts/verify_reward_ladder.py --tex writeup_interpscience.tex
else
  printf '%-46s%s\n' "reward ladder" "SKIP (tex or reward probe missing)"; skipped=$((skipped+1))
fi

# Table cells were gated long before prose numbers were, and the 2pp rewrite
# retyped ~20 figures into sentences where nothing checked them. This closes it.
for tex in writeup_interpscience; do
  if [ -f "$tex.tex" ] && [ -d followup/results/fragility/judge_lag ]; then
    run "prose numbers: $tex" \
        python -W ignore scripts/verify_prose_numbers.py --tex "$tex.tex"
  else
    printf '%-46s%s\n' "prose numbers: $tex" "SKIP (tex or scores missing)"; skipped=$((skipped+1))
  fi
done

# Everything else: headline accuracies, the arms and Table 2, Table 1, the
# probe baselines, the template score, run settings, Appendix F's quoted answer.
if [ -f writeup_interpscience.tex ]; then
  run "claims: writeup_interpscience" \
      python -W ignore scripts/verify_claims.py --tex writeup_interpscience.tex
fi

# Figures 1 and 2 are redrawn from the gated computations and compared byte for
# byte with the committed PDFs.
if [ -d followup/acts/phase0_harvest_runA/50 ]; then
  run "figures redraw identically" \
      python scripts/verify_figures.py
else
  printf '%-46s%s\n' "figures redraw identically" "SKIP (acts missing)"; skipped=$((skipped+1))
fi

echo
echo "=== coverage: is every number in the paper checked? ==="
run "every number claimed by a passing check" \
    python scripts/verify_numbers.py --tex writeup_interpscience.tex
run "planted errors are all caught" \
    python scripts/mutation_test.py --tex writeup_interpscience.tex

echo
echo "=== submission: is the thing you are about to upload safe? ==="
# writeup_interpscience is the de-anonymised camera-ready now (accepted
# 2026-09-29), so the anonymity scan would rightly fail on it. It gets the
# inverse check instead: real author block, final-mode style, no blind-review
# wording, the rollout files' structure, and no open TODO.
if [ -f writeup_interpscience.tex ] && [ -d followup/acts/phase0_harvest_runA/50 ]; then
  run "camera-ready: writeup_interpscience" \
      python -W ignore scripts/verify_camera_ready.py --tex writeup_interpscience.tex
else
  printf '%-46s%s\n' "camera-ready: writeup_interpscience" "SKIP (tex or acts missing)"; skipped=$((skipped+1))
fi

echo
# A skipped gate checked nothing, so it cannot count as a pass. Missing data is
# a failure unless ALLOW_SKIP=1 says the caller knows what they are not checking.
# A skipped gate also registers no claims, so verify_numbers.py fails on the
# numbers it owns: ALLOW_SKIP=1 gives a partial report, never a green suite.
if [ "$skipped" -gt 0 ] && [ "${ALLOW_SKIP:-0}" != "1" ]; then
  echo "$skipped gate(s) SKIPPED for missing inputs -- fetch them with"
  echo "  python scripts/artifacts.py fetch"
  echo "(or set ALLOW_SKIP=1 to accept a partial check)."
  fail=1
fi
if [ "$fail" -eq 0 ]; then
  echo "All gates pass."
else
  echo "SOMETHING DISAGREES OR WAS NOT CHECKED -- see above. Do not submit until this is green."
fi
exit "$fail"
