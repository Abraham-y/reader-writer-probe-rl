# overlap_split.json is retracted (2026-10-01)

`overlap_split.json` split the lag analysis into 248 "seen" and 158 "unseen"
prompts, on the belief that the reward probe was fit on 248 of the 406
evaluation prompts. It was not. The reward probe was fit on the first 300 rows
of the RL training pool (`eval_c_outcome_temp1_asingh_300.json`); their
`prompt_idx` values number that file, and the script compared them against the
evaluation file's indices. Matched by content (sorted numbers and target), the
overlap is zero: every one of the 406 evaluation prompts is unseen by the probe.

The "seen/unseen" split was therefore index < 300 vs index >= 300, which means
nothing. The paper's Appendix F and Table 5, built on it, were removed; the
script that produced this file was deleted. This file is kept, not deleted,
because `followup/results/` is append-only.

On all 406 prompts the full-population result stands: the probe's flag rate
moves at step 10 and its last-block AUROC at step 50.
