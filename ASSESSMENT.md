# Is the original paper worth submitting? Assessment, 2026-08-29

Short answer: **not as it stands, and for one specific fixable reason plus one
structural one.** But it is closer than the judge cut was, and the fix is mostly
promotion of things already in the repo rather than new work or new compute.

## The blocking problem

`writeup_judge.tex`'s abstract and Contribution 1 still say this:

> A linear correctness probe ... reaches held-out AUROC 0.982 and, as a
> best-of-16 selector, captures 88.8% of the headroom ... **Used as the RL
> reward it is catastrophic**

That is two different probes. Section 3's last sentence already retracts it:

> The probe promoted to reward ... is a temp-1 fit with held-out AUROC **0.810**
> --- not the 0.982 trace-final probe above, whose direction it shares only at
> cosine 0.163.

They differ in fit, in label rule (first-answer-block vs rollout-final), and in
sampling regime (temp 0.6/top-p 0.95 vs temp 1.0/top-p 1.0). The August audit
fixed the title and Section 3 and left the abstract and contribution list
carrying the identification, which is what a reviewer reads first.

So as written, the paper compares **probe X read-only** against **probe Y as
writer**, and never measures either in the other role. The reader/writer
asymmetry has no within-artifact evidence.

## The fix is already in the paper

Arm B is the experiment the paper needs and does not use as its headline:

| artifact | read-only AUROC | accuracy as RL reward |
|---|---|---|
| reward probe (runB) | 0.810 | 0.073 |
| Arm A, surface-residualised | 0.834 | 0.168 |
| **Arm B, surface features only** | **0.925** | **0.0000** |

One frozen 39-feature artifact. Good enough read-only to be worth deploying.
Exactly zero as a reward, with 99.9% of rollouts still emitting well-formed
answer blocks. That is the asymmetry, in a single artifact, already measured.

And the ladder is **non-monotone in monitor quality** -- a better read-only
monitor does not give a better outcome as a writer. That is a stronger claim
than the one currently in the abstract.

Caveat to state if you use it: the three read-only AUROCs are not commensurable.
0.834 and 0.925 are on the temp-0.6 clean-406 cache with first-block labels;
0.810 is temp-1.0 with rollout-final labels. Three estimators, one column.

## Other things that must change

- **Section 4.2 is superseded by the 29 Aug analysis.** It concludes "the judge
  stayed a good judge". The corrected finding is that the judge is a fixed
  function (965 repeated pairs, one changed verdict) and its flatness was never
  evidence about its health. Port that in or cut the section.
- **Flag rate "tightening steadily thereafter" is false on the paper's own
  Table 2**: it bottoms at 0.294 (step 80) and retraces to 0.317 while accuracy
  keeps falling.
- **The probe's 40-step flag-rate lead is partly definitional.** The flag rate is
  a threshold-crossing rate on the probe score, and the probe score is the RL
  objective. The judge case is the load-bearing one; say so.
- **Step 20 does not survive correction** across eleven checkpoints.
- **Limitations sits on page 7** of a 6pp cut. Desk-reject risk.
- **The interpretability content is cut.** The full version's "the probe is a
  reader, not a controller" section -- steering with matched random controls,
  the 0.939 position-stratified rebuttal to "it's a length detector" -- appears
  zero times in the 6pp. That matters for venue choice below.

## Venue

`STATUS.md` had two facts wrong. Corrected:

| Workshop | Deadline | Limit | Fit |
|---|---|---|---|
| **InterpScience** | **Sep 1** (tracker says Sep 2 -- confirm) | 5pp or 9pp, refs+appendices excluded | **Best.** Topics include "measurement validity ... and evaluation design" and "experimental designs that distinguish mechanisms from artifacts". That is this paper. No per-author cap, so the SAE paper does not block it. Forbids concurrent workshop submission. |
| Interp4Discovery | Sep 2 (extended) | 5pp main text | Partial. It wants interpretability used to *discover* things; this is interpretability used as a control signal. Forcing it is the same mistake that produced the judge paper. |
| ATTRIB | Sep 2 | -- | Partial: attributing model behaviour. |
| IAB / XAI4Science | Sep 6 | -- | Poor. Agent behaviour and scientific discovery respectively. |

Note the tension: InterpScience is the best fit *and* an adversarial reviewer's
first attack is that the 6pp cut has no interpretability left in it. If you go
there, restore the reader-not-controller section. At 9pp you have the room.

## Score

As submitted to an interpretability workshop: **3-4/10, reject** -- the abstract
asserts something the paper retracts, the interp content is cut, 4.2 is stale,
and it overflows.

After the fixes above, at InterpScience: **6-7/10**. Every ingredient exists in
this repo. None of it needs new compute.

## The honest summary

The judge paper was convoluted because it was built backwards from a venue. This
paper is not that -- it has a real thesis and hedges its premise properly
("we claim neither mechanism as novel"). What it has instead is an abstract that
was never updated after the August correction, and a better version of its own
headline sitting unused in Section 4.3.
