"""The ledger that makes "every number in the paper is checked" a tested fact.

WHY THIS EXISTS
The gates used to test whether a recomputed value appeared ANYWHERE in the
paper. A 2026-10-10 mutation test planted 23 wrong numbers in a copy of the tex
and every gate passed: a check for 0.755 was satisfied by a CI bound in another
table, a swap of two cells in Table 2 went unnoticed, and about 60 prose numbers
and 96 table cells were never compared at all.

Now every check that matches a number in the paper registers the exact pattern
it matched (the number in its context). `verify_numbers.py` runs last, extracts
every number token from the paper, and fails on any token that no registered
pattern claims and no listed exemption covers. A changed number then either
breaks the pattern of the check that owns it or is left unclaimed, so it fails
either way; `mutation_test.py` demonstrates that over many planted errors.

Usage inside a gate:

    import paper_ledger as L
    text = L.canonical(open(tex).read())        # the text every pattern is matched against
    L.claim("judge AUROC, step 99", r"step 99 \\(0\\.769")   # after the check passed
    ...
    L.dump("verify_camera_ready")               # writes $LEDGER_DIR/<name>.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re

LEDGER_DIR = os.environ.get("LEDGER_DIR", "/tmp/_ledger")

# Regions whose digits are not claims: cross-references, citations, URLs,
# layout commands, code identifiers in \texttt{}, and comments.
_EXCLUDE = [
    r"\\(?:ref|label|cite[tp]?|url|includegraphics|setlength|begin|end|hspace|vspace|renewcommand|arraystretch)\*?(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?",
    r"\\multicolumn\{\d+\}\{[^}]*\}",
    r"\\texttt\{[^}]*\}",
    r"\\tabcolsep\}\{[^}]*\}",
    r"\[leftmargin=[^\]]*\]",        # list-environment options
    r"\\\\\[\d+pt\]",            # \\[2pt] line spacing in the judge prompt
]
# A number token: optional sign, digits with optional {,} thousands groups, optional
# decimals. Digits right after a letter count too (L16, \sim91, \times10): an
# earlier rule skipped them, which left real statistics outside the ledger.
TOKEN = re.compile(r"(?<![\d.])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?")


def canonical(tex: str) -> str:
    """The paper's checked text: abstract to end, bibliography removed, TeX
    comments and excluded regions blanked to spaces (so offsets stay stable),
    and the math markup flattened the way the gates have always written numbers
    ($ dropped; {-}, {+}, {=} and \\, normalised)."""
    body = tex[tex.index(r"\begin{abstract}"):]
    a, b = body.index(r"\begin{thebibliography}"), body.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")
    body = body[:a] + re.sub(r"[^\n]", " ", body[a:b]) + body[b:]      # keep newlines: line numbers stay right
    body = re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), body)
    for pat in _EXCLUDE:
        body = re.sub(pat, lambda m: " " * len(m.group(0)), body)
    out = (body.replace("$", "").replace("{-}", "-").replace("{+}", "+")
               .replace("{=}", "=").replace("\\,", ""))
    global _text_sha
    _text_sha = hashlib.sha256(out.encode()).hexdigest()
    return out


def tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN.finditer(text)]


_claims: list[dict] = []
_text_sha: str | None = None   # the canonical text the claims were made against


def claim(label: str, pattern: str, ok: bool = True, setting: bool = False) -> None:
    """Record that a check matched `pattern` (a regex over canonical text) and
    whether its value agreed. Every number token inside a match is claimed.
    setting=True marks a run setting that recurs throughout the paper (the 406
    prompts, the 39 features): its pattern may pin context with a lookahead,
    because no other number in the paper could be swapped into its place."""
    _claims.append({"label": label, "pattern": pattern, "ok": bool(ok), "setting": bool(setting)})


def number_pattern(written: str) -> str:
    """Regex for one number token exactly as written, not inside a longer number."""
    return r"(?<![\d.])" + re.escape(written) + r"(?![\d])"


def dump(name: str) -> None:
    """Write this gate's claims, stamped with the text they were checked against,
    so verify_numbers.py can refuse a ledger left over from another version."""
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(os.path.join(LEDGER_DIR, f"{name}.json"), "w") as f:
        json.dump({"text_sha256": _text_sha, "claims": _claims}, f, indent=0)


def load_all(directory: str = LEDGER_DIR) -> list[dict]:
    out = []
    for fn in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if fn.endswith(".json"):
            d = json.load(open(os.path.join(directory, fn)))
            for c in d["claims"]:
                c["gate"] = fn[:-5]
                c["text_sha256"] = d["text_sha256"]
                out.append(c)
    return out


def coverage(text: str, claims: list[dict], exemptions: list[dict]):
    """(claimed token spans, unclaimed tokens, claim patterns that no longer match)."""
    covered = [False] * len(text)
    dead = []
    for c in list(claims) + list(exemptions):
        hits = list(re.finditer(c["pattern"], text, flags=re.S))
        if not hits:
            dead.append(c)
        for m in hits:
            for i in range(m.start(), m.end()):
                covered[i] = True
    unclaimed = [(s, e, t) for s, e, t in tokens(text) if not all(covered[s:e])]
    return unclaimed, dead
