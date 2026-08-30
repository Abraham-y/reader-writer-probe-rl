# STATUS — read this first

Single entry point. Everything else is either the deliverable, an active work
item, or archive. Last updated 2026-08-14.

---

## What this project is

A linear probe on Qwen2.5-0.5B layer-16 `</think>` hidden states predicts whether
a Countdown rollout will be correct. The finding is an asymmetry:

- **As a read-only selector it is excellent.** Best-of-16 by probe score reaches
  0.6700 against a random-pick-of-16 baseline of 0.5531 and an oracle ceiling of
  0.6847 — **89% of all available headroom**. It holds AUROC 0.939 with trace
  length stratified out, against 0.564 for a pure length detector.
  *(Corrected 2026-08-08: previously 0.6798 / 96% / 0.981. 0.6798 does not
  reproduce — the held-out figure is 0.6700 — the 0.5493 baseline came from a
  different rollout sample, and 0.981 was an in-sample number. See
  `REVISION_PACK.md` §B1–B2; all of it now comes out of
  `extension/probe/structural_baselines.py`.)*
- **As an RL reward it is catastrophic.** Probe score rises while true accuracy
  falls ~31 pp.

Same artefact, opposite outcomes. Reading a representation and writing to it are
different privileges, and probe accuracy licenses only the first.

**Target:** JUDGe workshop (backup: Verify-Agents). Deadline **2026-08-29**.

---

## Where it stands — 2026-08-14

**The paper is finished except for prose.** Every number is computed, verified,
and gated; every claim traces to a script that prints the published value beside
the recomputed one and exits non-zero on disagreement.

### Venue

**Not submitted to JUDGe (deadline 2026-08-29).** Decided against on the night.
The finding is sound and every number is gated, but the paper around it was not
finished, and the reasons were specific rather than nerves:

1. **Three actors, never distinguished.** The policy (Qwen 0.5B, the thing that
   broke), the probe (the bad reward that broke it), and the judge (Qwen 7B,
   which never changed) are all called "the model" at various points. A reader
   meeting this cold cannot follow it. A table of the three would fix it.
2. **The framing is buried.** The whole exercise is only possible because
   Countdown has an exact checker: you cannot compute a judge's AUROC without
   ground truth, so nobody monitors it in production, they compute it once at
   validation and file it. That is the frame for everything and it currently
   appears on page 2 as an aside.
3. **The premise is overstated.** "Teams just monitor judge quality" is half a
   straw man. People do watch score distributions; Li (2026) exists because they
   do and cannot attribute the drift. The honest contribution is narrower:
   what that ordinary alarm buys you, in RL steps, against a monitor whose own
   report card still reads healthy.

Also unresolved: the prose is largely LLM-written and reads that way. NeurIPS
policy permits this (authors own the content, no disclosure needed for writing
aid) but it is not something to present from a stage.

**Where it can still go.** All NeurIPS 2026, later deadlines, all needing only
modest reframing:

| Workshop | Deadline |
|---|---|
| ATTRIB (Attributing Model Behavior at Scale) | Sep 2 |
| Interp4Discovery | Sep 3 |
| IAB (Interpreting Agent Behavior) | Sep 6 |
| XAI4Science | Sep 6 |

InterpScience (Sep 2) is spoken for by the SAE paper and forbids concurrent
submission.

### What the paper actually found

A fixed judge, fed a policy collapsing from 54.4% to 22.6% accuracy, reports an
essentially unchanged AUROC (0.785 -> 0.769). Not because it has a blind spot:
because it is a fixed function and was never the variable. Of 965 identical
(problem, equation) pairs seen early and late, exactly ONE gets a different
verdict. What changed is the input distribution, in composition and in surface
form -- the policy stops writing parentheses (0.017 -> 0.982 of correct answers)
and the judge treats the same arithmetic 20 points differently depending on
bracketing (+0.198, 95% CI [+0.131, +0.265]).

Precision looks like it collapses (0.746 -> 0.429) but that is the same
confound in reverse. The one statistic that moves and needs no labels is the
flag rate, 0.500 -> 0.317.

### The one open item

**Rewrite the prose in your own voice.** The current draft is assembled from many
surgical edits and reads like it. Two specific things worth fixing while you do:

1. **The spine and the title disagree.** The paper is titled around the lag but
   still opens as the reader/writer paper, with the lag as a subsection. Making
   the lag the argument — and the reader/writer asymmetry the setup that makes it
   measurable — is the restructure, and it is a judgement about the thesis.
2. **Section 3 shows its seams**, having been compressed from a much longer
   version.

**After any rewrite, run `bash scripts/check_everything.sh`.** Rewriting is when
numbers get retyped, and retyping is how every defect in the August audit got in.

### What is verified

Six gates, all green:

| gate | covers |
|---|---|
| `structural_baselines.py` | selection table, length stratification, dropped-rollout population |
| `surface_battery.py` | the surface decomposition and the frozen length detector |
| `quantify_structural_confound.py` | the template table |
| `verify_residual_arms.py` | both pre-registered arms + Arm B's output-shape claims |
| `verify_lag_result.py` | the 40-step lag, its scope condition, the control, the early-warning null |
| `make_submission_tex.py --check` | anonymity of the file actually being uploaded |

### Assessment

**≈8.0 — accept, poster.** Up from a reject-grade draft on 2026-08-12. Ceiling is
novelty (~6.1): the evasion mechanism is owned by Gupta & Jenner and the
Obfuscation Atlas, and what is ours — the sign of the objective, the lag, the free
decomposition instrument — is real but narrow. 0.5B and n=1 seeds are what a
reviewer would cite against a talk. Neither is buyable before the deadline.

### Deliberately not doing before 2026-08-29

- **A second training seed** (~$100). The +2.84 pp it would firm up is already
  reported against the baseline's own 4–6 pp checkpoint-to-checkpoint drift.
- **1.5B replication** (~$150–300, 2–3 days). This is the next paper, not this one.
- **Harvesting runB's ladder** (~$20). Checked and it would buy nothing: runB's
  monitor starts at AUROC 0.590, near chance, so there is no intact window for a
  lag to appear in. That fact is now *in* the paper as the scope condition.

### What changed to get here

 earlier (2026-08-08 audit)

- **§3 must be cut.** Its table claims the steered vector is the direction RL
  optimised at "cosine 1.000, AUROC 0.982". Measured: **cosine 0.163**, AUROC
  0.896. Steering was also applied one transformer block and 2–3 tokens
  downstream of the probe's read site (already flagged in `CODE_AUDIT.md:503-505`),
  and neither key contrast is significant (p = 0.063, p = 0.080). Not fixable —
  the checkpoints are gone.
- **§1 and §4 verify exactly.** The drift table reproduces to the digit
  (2983/490/150/1835); §4's +1.5 pp and 1.83→0.91 reproduce to the decimal.
- **Line 57 was wrong and its replacement is better.** "Statistically
  indistinguishable from the F→F floor" is false (+0.065, CI [+0.023, +0.105]),
  but the residue is only 8.7% of the way from floor to ceiling — a measurable,
  almost entirely erased trace. That is a positive result where there was a null.
- **Line 24's novelty claim is false.** Gupta & Jenner study exactly what it says
  nobody studies. Must be rewritten; the real delta is that prior work uses the
  monitor as a *penalty* (evasion is the argmax) while this uses it as a *positive
  reward* with no evasion incentive — and it breaks anyway.
- **New centrepiece (§G2).** A 7-feature logistic regression — no network, no
  forward pass — reproduces the probe's Goodhart signature: its score rises
  0.506 → 0.832 while true accuracy falls 0.498 → 0.130.
  *(Corrected 2026-08-08: previously 0.588 → 0.932 and 0.588 → 0.276. Until now
  no script in the repo computed §G2 at all — `n_backtrack` and `has_therefore`
  existed in no source file — so the old figures were unreproducible. It is now
  `extension/probe/surface_battery.py`; the claim holds, the magnitudes moved,
  and the per-feature shifts turn out to be negative, not positive.)*

Full detail and every corrected table: **`REVISION_PACK.md`**.

---

## The experiment

Testing the remediation the paper's own line 179 proposes as an open question:
does a probe that cannot read cheap surface features resist Goodharting?

| arm | reward probe | read-only AUROC | accuracy @ step 100 |
|---|---|---|---|
| reference | C_outcome (verifier RL) | — | 0.5306 |
| reference | published probe-as-reward | 0.978 | 0.2361 |
| **A** | surface-residualised `LR(h − s@B)` | 0.834 | **0.1678** |
| **B** | surface features ONLY, ignores activation | 0.925 | **0.0000** |

**Arm A falsified its own pre-registered prediction.** Residualising made the
collapse **worse**: −6.83 pp vs the published run, CI [−8.85, −4.79]. The
remediation the paper proposes does not work and backfires.

The "weaker probes collapse less" confound is dead — Arm A is weaker *and*
collapsed harder, so weakness cannot explain a result in that direction.

**Arm B CONFIRMED its pre-registered prediction.** A reward with no access to the
model's internals destroyed the task outright (0.0000, −23.6 pp vs the published
run, CI [−26.4, −20.9]). §G2 is now a tested claim. The failure is not degenerate:
99.9% of rollouts still emit `<answer>`, but the block contains reasoning prose
instead of an equation, and mean length doubled. See `REVISION_PACK.md` §G5.

Total cost ~$90. Both arms complete.

---

## Active files

| file | what it is |
|---|---|
| `writeup_judge.tex` / `.pdf` | **the submission** — anonymous, NeurIPS style, 6 pp main text |
| `writeup_workshop_full.tex` / `.md` | the long-form 15 pp version; source for the later conference paper |
| `scripts/check_everything.sh` | **run this after any edit** — all six gates in one command |
| `REVISION_PACK.md` | every corrected number, its population, and why the earlier value was wrong |
| `CODE_AUDIT.md` | correctness pass over the results path, with a status banner |
| `followup/.../PREREGISTRATION.md` | **do not edit** — predictions committed before the arms ran |
| `followup/RELATED_WORK.md` | literature position. Note: it is agent summaries, not primary reading, and it has been wrong twice (prompt counts, and a "126/300 split" that the manifest contradicts). Verify before quoting. |

## Reference (not working docs)

`findings.md`, `CODE_AUDIT.md`, `writeup.md`, `README.md`,
`RUNBOOK_CLEAN_REPLICATION.md`, `followup/CLAUDE.md`, `followup/README.md`,
`followup/PAPER_NOTES.md` (carries explicit retractions — worth reading before
re-asserting any earlier claim).

Superseded process docs are under `docs/archive/`.

---

## Two standing traps

1. **Do not let §G2 or §G4 become mechanism claims.** The measured facts are that
   a surface model reproduces the Goodhart signature, and that residualising
   coincided with a worse collapse. *Why* is unestablished. "Less-constrained
   boundary, more room to exploit" is a hypothesis. Asserting it as a finding is
   exactly how §3 became indefensible.
2. **"Statistically indistinguishable" has appeared three times in this paper
   without a test** (§3's null band, line 88, line 57). At least two were wrong.
   Search for the phrase before submitting.

## One methodological note worth keeping

Five deployment defects were hit during the experiment — credentials, Ray
`PYTHONPATH`, sklearn version, `load_dataset` on a local file, container path.
**All five lived outside what in-process gates can see, and the gates stayed green
and correct throughout.** In-process correctness and deployment correctness are
close to disjoint. A 5-step smoke run (~$2, ~10 min) caught every one of them
before a 5.5-hour job did.
