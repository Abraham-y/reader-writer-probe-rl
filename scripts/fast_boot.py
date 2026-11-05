"""Prompt-clustered paired bootstrap of AUROC and threshold rates, vectorised.

The analyses (changepoint_lag.py, analyze_judge_lag.py) resample prompts with
`rng.choice(prompts, len(prompts))` and recompute each statistic on the
concatenated rows. Drawing the same prompts and weighting each row by how many
times its prompt was drawn gives the same statistic exactly (a row drawn k times
counts k times in the Mann-Whitney sums, ties at one half), so this module
reproduces their seed-0 intervals and p-values draw for draw while running fast
enough to pool several seeds. Pooling matters only where a p-value sits near a
correction threshold: there the Monte Carlo error of one 2,000-draw run decides
the verdict (the judge's step-30 dip was p = 0.002 at seed 0, 0.002-0.006 over
seeds, against a Bonferroni cut-off of 0.0025).
"""
from __future__ import annotations

import numpy as np


def draws(prompts: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_prompts) counts, from the same RNG calls as the analyses."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, len(prompts)), dtype=np.int32)
    for b in range(n_boot):
        d = rng.choice(prompts, size=len(prompts), replace=True)
        out[b] = np.bincount(np.searchsorted(prompts, d), minlength=len(prompts))
    return out


class Checkpoint:
    """Scores, labels and prompt index of one checkpoint's rows."""

    def __init__(self, scores, labels, prompt_idx, prompts):
        o = np.argsort(scores, kind="mergesort")
        self.s = np.asarray(scores, float)[o]
        self.y = np.asarray(labels, int)[o]
        self.col = np.searchsorted(prompts, np.asarray(prompt_idx)[o])
        # groups of tied scores, for the half-credit in Mann-Whitney
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.s)) + 1]

    def weights(self, counts: np.ndarray) -> np.ndarray:
        return counts[:, self.col].astype(np.float64)

    def auroc(self, counts: np.ndarray) -> np.ndarray:
        w = self.weights(counts)
        wp = np.add.reduceat(w * self.y, self.starts, axis=1)
        wn = np.add.reduceat(w * (1 - self.y), self.starts, axis=1)
        below = np.cumsum(wn, axis=1) - wn
        num = (wp * (below + 0.5 * wn)).sum(1)
        return num / (wp.sum(1) * wn.sum(1))

    def rate(self, counts: np.ndarray, thr: float, which: str = "all") -> np.ndarray:
        w = self.weights(counts)
        m = (self.s >= thr).astype(float)
        if which == "pos":
            w = w * self.y
        elif which == "neg":
            w = w * (1 - self.y)
        return (w * m).sum(1) / w.sum(1)


def p_two_sided(d: np.ndarray, strict: bool = True) -> float:
    """2 * the smaller tail. strict=True counts d < 0 and d > 0 (the AUROC
    convention in changepoint_lag.py); False counts d <= 0 and d >= 0."""
    if strict:
        return float(2 * min((d < 0).mean(), (d > 0).mean()))
    return float(2 * min((d <= 0).mean(), (d >= 0).mean()))


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm step-down: which hypotheses are rejected."""
    order = sorted(pvals, key=pvals.get)
    out, m, stop = {}, len(pvals), False
    for i, k in enumerate(order):
        if not stop and pvals[k] <= alpha / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out
