#!/usr/bin/env python3
"""Redraw both figures and fail unless they are byte-identical to the committed ones.

The two plot scripts draw from the computations the gates check:
plot_ladder_from_gated.py from verify_reward_ladder.arm_aurocs and the arms gate,
plot_lag_from_gated.py from the change-point files that verify_tables.py checks
against its own recomputation. Both save with a fixed creation date, so a redraw
from unchanged inputs reproduces the committed PDF exactly. A figure edited by
hand, or left stale after its inputs changed, fails here.

Byte equality assumes the pinned matplotlib (requirements-regenerate.txt); a
different version embeds a different Producer string and fonts.

    python scripts/verify_figures.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES = (("figures/lag.pdf", "scripts/plot_lag_from_gated.py"),
           ("figures/ladder.pdf", "scripts/plot_ladder_from_gated.py"))


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for fig, script in FIGURES:
            out = os.path.join(tmp, os.path.basename(fig))
            r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(_ROOT, script), "--out", out],
                               cwd=_ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                bad.append(f"{script} failed:\n{r.stderr[-800:]}")
                continue
            same = sha(out) == sha(os.path.join(_ROOT, fig))
            print(f"{fig:22s} {'identical to the redraw' if same else 'DIFFERS from the redraw'}")
            if not same:
                bad.append(f"{fig} is not what {script} draws; regenerate it with: python {script} --out {fig}")
    if bad:
        print(f"\nFAIL ({len(bad)})")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("Both figures are redrawn exactly from the gated computations.")


if __name__ == "__main__":
    main()
