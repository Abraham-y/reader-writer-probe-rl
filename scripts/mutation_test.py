#!/usr/bin/env python3
"""Plant errors in the paper and require that the gates catch every one.

Runs after verify_numbers.py, on the claims the gates registered (paper_ledger).
Two kinds of error are planted, one at a time, into the text the gates read:

  * every number token that is not exempt gets one digit changed, which is the
    retyped-number error the 2026-10-10 audit found the old gates passing;
  * pairs of nearby numbers with the same format are swapped, which a check
    that tests only whether a value appears somewhere cannot see (the old
    reward-ladder gate passed a swap of arm A's and arm B's AUROCs).

An error is caught when some passing claim's pattern stops matching or some
number is left unclaimed, which makes verify_numbers.py fail. Recomputation is
not repeated: a planted error changes only the text, so a claim that still
matches would still pass. Exempt numbers (scripts/number_exemptions.tsv) are not
planted, because by definition nothing regenerates them.

    python scripts/mutation_test.py [--tex writeup_interpscience.tex] [--swaps N]   # default: every pair
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ledger as L  # noqa: E402
from verify_numbers import exemptions  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def caught(text, claims, ex) -> bool:
    unclaimed, dead = L.coverage(text, claims, ex)
    return bool(unclaimed or dead)


def exempt_spans(text, ex):
    spans = []
    for c in ex:
        spans += [(m.start(), m.end()) for m in re.finditer(c["pattern"], text, flags=re.S)]
    return spans


def bump(tok: str) -> str:
    """Change the last digit of a number token, keeping its format."""
    i = max(j for j, ch in enumerate(tok) if ch.isdigit())
    return tok[:i] + str((int(tok[i]) + 1) % 10) + tok[i + 1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    ap.add_argument("--swaps", type=int, default=0, help="0 = every eligible pair")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    text = L.canonical(open(os.path.join(_ROOT, a.tex)).read())
    claims = [c for c in L.load_all() if c["ok"]]
    ex = exemptions()
    if caught(text, claims, ex):
        sys.exit("the unmutated paper does not pass the ledger; run verify_numbers.py first")
    ex_sp = exempt_spans(text, ex)
    toks = [t for t in L.tokens(text) if not any(s <= t[0] and t[1] <= e for s, e in ex_sp)]

    missed = []
    for s, e, tok in toks:
        if not caught(text[:s] + bump(tok) + text[e:], claims, ex):
            missed.append(f"changed {tok!r} -> {bump(tok)!r} near: ...{' '.join(text[max(0, s - 50):e + 30].split())}...")
    print(f"single-digit changes: {len(toks) - len(missed)}/{len(toks)} caught")

    rng = random.Random(a.seed)
    shape = lambda t: re.sub(r"\d", "9", t)
    pairs = [(x, y) for i, x in enumerate(toks) for y in toks[i + 1:i + 12]
             if y[0] - x[1] < 200 and shape(x[2]) == shape(y[2]) and x[2] != y[2]]
    rng.shuffle(pairs)
    sw_missed = 0
    pairs = pairs[:a.swaps] if a.swaps else pairs
    for (s1, e1, t1), (s2, e2, t2) in pairs:
        mutant = text[:s1] + t2 + text[e1:s2] + t1 + text[e2:]
        if not caught(mutant, claims, ex):
            sw_missed += 1
            missed.append(f"swapped {t1!r} and {t2!r} near: ...{' '.join(text[max(0, s1 - 40):e2 + 20].split())}...")
    n_sw = len(pairs)
    print(f"swaps of nearby same-format numbers: {n_sw - sw_missed}/{n_sw} caught")
    if missed:
        print(f"\nFAIL: {len(missed)} planted errors pass every gate")
        for m in missed[:60]:
            print("  " + m)
        sys.exit(1)
    print("Every planted error is caught.")


if __name__ == "__main__":
    main()
