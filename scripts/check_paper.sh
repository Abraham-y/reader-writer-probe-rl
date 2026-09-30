#!/usr/bin/env bash
# Fast feedback loop while rewriting prose. Rebuilds one paper and answers the
# four questions a prose edit can get wrong:
#
#   does it still compile, does it still FIT the page limit, is it still
#   anonymous, and did any AI-writing tell creep back in.
#
#   bash scripts/check_paper.sh                      # the camera-ready, 9pp
#
# This does NOT re-verify the table numbers -- that needs bootstraps and takes
# a few minutes. Run `bash scripts/check_everything.sh` before you submit.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TEX="${1:-writeup_interpscience}"
LIMIT="${2:-9}"

pdflatex -interaction=nonstopmode "$TEX.tex" >/dev/null 2>&1
pdflatex -interaction=nonstopmode "$TEX.tex" >/tmp/_cp.log 2>&1

if [ ! -f "$TEX.pdf" ]; then
  echo "BUILD FAILED -- LaTeX error:"
  grep -A3 "^!" /tmp/_cp.log | head -20
  exit 1
fi

python3 - "$TEX" "$LIMIT" <<'PY'
import re, subprocess, sys
tex, limit = sys.argv[1], int(sys.argv[2])
pages = subprocess.run(["pdftotext", tex + ".pdf", "-"],
                       capture_output=True, text=True).stdout.split("\f")
# Body runs until the References heading, wherever that falls. It may begin
# partway down a page and the bibliography may then run past the limit, which
# is fine: the limit applies to body pages only. The earlier version of this
# check assumed references always started AFTER the limit and reported a false
# overflow when they did not.
ref_page = next((i for i, pg in enumerate(pages) if "References" in pg), len(pages) - 1)
# Body may end BEFORE the references page if References start at its very top.
_head = pages[ref_page][:pages[ref_page].index("References")] if ref_page < len(pages) else ""
_has_body = bool([l for l in _head.splitlines()
                  if l.strip() and not re.fullmatch(r"[\d\s]+", l.strip())])
body_end = ref_page + 1 if _has_body else ref_page
if body_end <= limit:
    spill = []
else:
    over = pages[limit] if len(pages) > limit else ""
    body = over[:over.index("References")] if "References" in over else over
    spill = [l for l in body.splitlines()
             if l.strip() and not re.fullmatch(r"[\d\s]+", l.strip())]

src = open(tex + ".tex").read()
prose = src[src.index(r"\begin{abstract}"):src.index(r"\begin{thebibliography}")]
log = open("/tmp/_cp.log").read()

print(f"  build          OK ({len(pages)-1} pages)")
if spill:
    print(f"  PAGE LIMIT     OVER by {len(spill)} lines on p{limit+1}")
    print(f"                 first spilled: {' '.join(spill[0].split())[:70]}...")
else:
    print(f"  page limit     OK (body ends p{body_end}, limit {limit}pp)")

warn = len(re.findall(r"Overfull|Undefined", log))
print(f"  latex warnings {'OK' if warn==0 else str(warn)+'  <-- check /tmp/_cp.log'}")

# em dashes: one is expected (the table's empty-cell marker)
em = len(re.findall(r"---", prose))
print(f"  em dashes      {em}" + ("  (just the table marker)" if em <= 1
      else "  <-- the main AI tell; aim for 0 in prose"))
flat = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", prose, flags=re.S)
flat = re.sub(r"[\\{}$]", " ", re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", flat)).replace("\n", " ")
longs = [s for s in re.split(r"(?<=[.!?]) ", flat) if len(s.split()) > 45]
print(f"  45+ word sents {len(longs)}" + ("" if not longs else f"  <-- e.g. {' '.join(longs[0].split())[:60]}..."))

# The rest of the AI-writing tells, so a rewrite cannot quietly reintroduce
# them. Numeric ranges (1--3) and the table's empty-cell marker are not dashes
# used as prose asides, so they are excluded rather than flagged forever.
TELLS = {
    "AI vocabulary": r"\b(crucial|delve|pivotal|underscore\w*|showcase\w*|testament|tapestry|landscape|intricate|nuanced|realm|foster\w*|garner\w*|leverage\w*|seamless\w*|vibrant|profound)\b",
    "copula avoidance": r"\b(serves as|stands as|represents a|acts as a|functions as)\b",
    "negative parallel": r"(not (just|merely|only) \w+[^.]{0,40}\bbut\b)",
    "authority trope": r"\b(the real question is|at its core|fundamentally|what really matters)\b",
    "signposting": r"\b(let us|let's|we now turn|without further ado)\b",
    "filler": r"\b(it is important to note|in order to|due to the fact that)\b",
    "curly quotes": "[\u201c\u201d\u2018\u2019]",
}
found = {n: re.findall(p, flat, re.I) for n, p in TELLS.items()}
found = {n: h for n, h in found.items() if h}
if not found:
    print("  ai-writing tells 0")
else:
    for n, h in found.items():
        ex = h[0] if isinstance(h[0], str) else h[0][0]
        print(f"  ai-writing tells {n}: {len(h)}  <-- e.g. \"{ex}\"")

words = [len(x.split()) for x in re.split(r"(?<=[.!?]) ", flat) if x.strip()]
if words:
    print(f"  rhythm         {len(words)} sentences, mean {sum(words)//len(words)}w, "
          f"{sum(1 for w in words if w < 12)} under 12w")
PY

# A camera-ready is de-anonymised on purpose; scanning it for names would only
# ever fail. scripts/verify_camera_ready.py checks its front matter instead.
if grep -q '\\usepackage\[final\]' "$TEX.tex"; then
  echo "  anonymity      n/a (camera-ready; see scripts/verify_camera_ready.py)"
elif python3 scripts/make_submission_tex.py --check "$TEX.tex" >/dev/null 2>&1; then
  echo "  anonymity      OK"
else
  echo "  anonymity      FAIL -- real name or repo URL leaked:"
  python3 scripts/make_submission_tex.py --check "$TEX.tex" 2>&1 | tail -5 | sed 's/^/                 /'
fi
