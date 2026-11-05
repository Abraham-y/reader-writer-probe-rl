#!/usr/bin/env python3
"""Fail unless every number in the paper is claimed by a passing check.

Runs last in check_everything.sh. Each gate before it registers, through
paper_ledger.claim, the pattern it matched in the paper: a number together with
enough of its sentence to pin it to one place. This script

  1. extracts every number token from the paper (abstract to end, bibliography
     and cross-references excluded; see paper_ledger.canonical),
  2. marks the tokens inside any passing claim's matches, or inside a listed
     exemption (scripts/number_exemptions.tsv: cited results, the withdrawn
     steering numbers, names and dates, each with its reason),
  3. fails on any unclaimed token, on any claim whose check failed, on any claim
     pattern that no longer matches, and on any claim that pins its number with
     too little context to tell it from another occurrence of the same digits.

A changed number is then caught twice: the check that owns it stops matching,
and the new token is left unclaimed. `mutation_test.py` plants errors to show it.

    python scripts/verify_numbers.py [--tex writeup_interpscience.tex]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ledger as L  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXEMPT = os.path.join(_ROOT, "scripts", "number_exemptions.tsv")
MIN_CONTEXT = 6     # characters of non-numeric context a claim must carry
MAX_SPAN = 600      # a claim longer than this is too broad to pin anything


def exemptions() -> list[dict]:
    out = []
    for line in open(EXEMPT):
        if not line.strip() or line.startswith("#"):
            continue
        pattern, reason = line.rstrip("\n").split("\t", 1)
        out.append({"label": "exempt: " + reason, "pattern": pattern, "ok": True, "gate": "exemptions"})
    return out


def context_ok(text: str, m: re.Match) -> bool:
    """A claim must match more than its number: at least MIN_CONTEXT characters
    beyond its longest token (words, symbols, or a neighbouring number), so that
    the same digits elsewhere in the paper cannot satisfy it."""
    s = m.group(0)
    longest = max((len(t.group(0)) for t in L.TOKEN.finditer(s)), default=0)
    return len(s.strip()) - longest >= MIN_CONTEXT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    ap.add_argument("--ledger", default=L.LEDGER_DIR)
    a = ap.parse_args()
    tex = open(os.path.join(_ROOT, a.tex)).read()
    text = L.canonical(tex)
    base = tex[:tex.index(r"\begin{abstract}")].count("\n")
    claims = L.load_all(a.ledger)
    ex = exemptions()
    bad = []

    stale = sorted({c["gate"] for c in claims if c.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest()})
    for g in stale:
        bad.append(f"ledger of {g} was made against a different version of the paper; rerun it")
    failed = [c for c in claims if not c["ok"]]
    for c in failed:
        bad.append(f"check failed, so its number is not verified: [{c['gate']}] {c['label']}")
    good = [c for c in claims if c["ok"]]
    for c in good:
        hits = list(re.finditer(c["pattern"], text, flags=re.S))
        if any(len(m.group(0)) > MAX_SPAN for m in hits):
            bad.append(f"claim too broad (> {MAX_SPAN} chars): [{c['gate']}] {c['label']}")
        if hits and not c.get("setting") and not all(context_ok(text, m) for m in hits):
            bad.append(f"claim pins its number with too little context: [{c['gate']}] {c['label']}  /{c['pattern']}/")
    unclaimed, dead = L.coverage(text, good, ex)
    for c in dead:
        bad.append(f"pattern no longer matches the paper: [{c['gate']}] {c['label']}  /{c['pattern']}/")

    toks = L.tokens(text)
    print(f"{len(toks)} number tokens; {len(toks) - len(unclaimed)} claimed by {len(good)} checks "
          f"from {len({c['gate'] for c in claims})} gates and {len(ex)} exemptions")
    for s, e, t in unclaimed:
        line = text.count("\n", 0, s) + 1
        ctx = " ".join(text[max(0, s - 60):e + 40].split())
        bad.append(f"unclaimed number {t!r} near paper line {line + base}: ...{ctx}...")
    if bad:
        print(f"\nFAIL ({len(bad)})")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print("Every number in the paper is claimed by a passing check or a listed exemption.")


if __name__ == "__main__":
    main()
