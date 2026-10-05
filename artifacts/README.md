---
license: cc-by-4.0
pretty_name: "What Happens to a Monitor's Accuracy When You Train Against It: cached artifacts"
language:
- en
tags:
- interpretability
- probing
- reward-hacking
- llm-as-a-judge
- reinforcement-learning
viewer: false
---

# Cached artifacts for "What Happens to a Monitor's Accuracy When You Train Against It"

Abraham Yeung and Anagha Ramaswamy, Stanford University.
Interpretability as a Science (InterpScience) Workshop, NeurIPS 2026.
Code and paper: <https://github.com/Abraham-y/reader-writer-probe-rl>

These are the 65 files (0.36 GB) that the paper's regeneration script reads. With
them, every number, table and figure in the paper regenerates on CPU, and the
script exits non-zero if any published value disagrees with its recomputation.
The list is not hand-picked: it was built by logging every file the script
opened, and it was then checked by running the full script in an empty checkout
that held only the repository and these files.

## Use

The dataset mirrors the code repository's paths, so the files land where the
scripts look for them.

```bash
git clone https://github.com/Abraham-y/reader-writer-probe-rl
cd reader-writer-probe-rl
pip install huggingface_hub   # plus the repository's own requirements
python scripts/artifacts.py fetch
bash scripts/check_everything.sh
```

`fetch` downloads exactly the paths listed in `artifacts/MANIFEST.tsv` and then
checks each file's size and SHA-256 against it. It deliberately does not
download this card, which would otherwise overwrite the code repository's
README. To check a copy you already have: `python scripts/artifacts.py verify`.

## Contents

All activations are the layer-16 residual stream (`hidden_states[16]`) of a
Qwen2.5-0.5B policy at the first token of `</think>`, stored as float32 arrays
of shape (answers, 896). "First block" and "last block" grade the first and the
last `<answer>` block an answer contains, with the exact Countdown verifier.

| path | what it is | where the paper uses it |
|---|---|---|
| `followup/acts/phase0_harvest_runA/{step}/16.npy`, `labels.parquet` | activations and labels for 8 fresh answers per prompt, sampled at temperature 1 on the 406 evaluation prompts, at each of the 11 checkpoints of `runA` (the probe-as-reward run from C_outcome). Labels hold prompt and answer indices, both block labels, answer length and template features | §4.1, Figure 1, Tables 3 and 4 |
| `followup/acts/phase0_harvest_runA__fixed_text/40/16.npy`, `labels.parquet` | the step-0 answers of that ladder, forward-passed through the step-40 checkpoint, with their step-0 labels | §4.1 and Appendix B: the score inflation is in the policy's new answers, not in how the probe reads old ones |
| `followup/results/fragility/judge_lag/step_{step}.jsonl` | an independent draw of 8 answers per prompt at each `runA` checkpoint: the proposed equation, the verifier's first-block label, and `Qwen2.5-7B-Instruct`'s renormalised P(YES) | §4.2, Table 5 |
| `followup/acts/vanilla_rloo_ladder/{step}/16.npy`, `labels.parquet` | the same kind of activations for five checkpoints of ordinary verifier-reward RLOO: the weak control ladder the paper describes and does not rely on | Appendix B |
| `extension/cache/probe_cache_n500_clean406/*` | activations (`.npz`) and first-block labels (`.meta.json`) for C_outcome's and C_SFT's headline-protocol answers (16 per prompt, temperature 0.6) | §3, the AUROC column of Table 2, and the rewards' AUROCs read through C_SFT (§4.3) |
| `extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer_temp1.pkl` | the reward probe: a scikit-learn `StandardScaler` + `LogisticRegression` pipeline | the RL reward; scored in §4.1 |
| `eval_c_outcome_temp1_asingh_300.json` | the reward probe's fitting data: 8 temperature-1 answers to each of 300 prompts from the RL training pool | the check that none of those prompts is among the 406 (§3) |
| `extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl` | the trace-final probe (AUROC 0.982) | its cosines with the reward probe and with arm R's probe (§3) |
| `followup/experiments/fragility/residual_probe/probe_surface_only.pkl` | arm B's reward, the 39-feature surface model | its weights and AUROCs (§4.3, Table 2) |
| `followup/experiments/fragility/residual_probe/probe_surface_residual_l16.pkl` | arm A's reward: the surface-residualised probe `LR(h - s B)` | its AUROC (Table 2), its split into an activation term and a text term, and its AUROC read through C_SFT (§4.3) |
| `followup/experiments/fragility/residual_probe/probe_raw_arm_recipe.pkl` | arm R's reward: arm A's probe fit without the residualisation (built by `build_raw_arm.py`) | its AUROCs (Table 2, §4.3) and its cosine with the trace-final probe (§3) |
| `eval_*.json` | sampled answers with verifier scores: `runA` and `runB` after training and C_outcome, C_SFT (headline protocol); arms A, B and R, and `runB` re-scored as arm C (arm protocol); `*_rescore.json` re-run arms A and B with the recorded arm-protocol command and reproduce their files answer for answer (Appendix C) | §4, §4.3, Table 2 |
| `artifacts/MANIFEST.tsv` | every file above with its size and SHA-256 | `scripts/artifacts.py verify` |

The probe files are Python pickles (the `.pkl` files, and `.npz` files loaded
with `allow_pickle=True`). Loading a pickle can execute code, so load these only from
this dataset or another source you trust.

## Provenance and licenses

- These artifacts are released under CC BY 4.0, the paper's license.
- The policies are fine-tunes of `Qwen/Qwen2.5-0.5B` (Apache-2.0), and the judge
  is `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0).
- C_SFT is `asingh15/qwen-sft-countdown-defaultproj`, a public
  checkpoint we did not train; its model card declares no license. This dataset
  contains its sampled answers and activations, not its weights or training data.
- Countdown problems are procedurally generated, following Gandhi et al. (2024),
  "Stream of Search".

## What is not here

- Model checkpoints. The RL checkpoints are not public; everything the paper
  reports is computed from the activations and answers above.
- Anything behind the withdrawn steering result, whose checkpoints no longer exist.
- Training variance: every RL configuration in the paper is a single seed.

## Citation

```bibtex
@inproceedings{yeung2026monitor,
  title     = {What Happens to a Monitor's Accuracy When You Train Against It},
  author    = {Yeung, Abraham and Ramaswamy, Anagha},
  booktitle = {Interpretability as a Science (InterpScience) Workshop at NeurIPS},
  year      = {2026}
}
```
