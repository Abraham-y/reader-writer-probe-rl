"""Edit the paper as plain text; rebuild the LaTeX from it.

WHY THIS EXISTS
Rewriting prose inside LaTeX means stepping around \\textbf, \\emph, escaped
percent signs and float placement while trying to think about sentences. This
lets the prose live in a .md file and rebuilds the .tex from it.

WHAT IT WILL NOT LET YOU BREAK
The table, the preamble and the bibliography are NEVER taken from the markdown.
They are lifted verbatim from the existing .tex every time, because the table's
66 cells are checked against recomputed data by verify_paper_tables.py and a
retyped digit there is exactly the defect that check exists to catch. The
markdown carries a {{TABLE}} marker; the real table is spliced in at build time.

    python scripts/md_to_tex.py --extract       # .tex -> .txt (once, to start)
    python scripts/md_to_tex.py                 # .txt -> .tex (after editing)

LIGHT MARKUP YOU CAN USE (everything else passes through as-is)
    ## Heading            -> \\section{Heading}
    **bold**              -> \\textbf{bold}
    *italic*              -> \\emph{italic}
    {{TABLE:tab:judge}}   -> that verified table, spliced from the .tex
    $x = 1$               -> passed through untouched
    \\citep{key}           -> passed through untouched
    ---                   -> converted to a comma; em dashes are the main AI tell

Anything else is passed through, so LaTeX you type by hand still works.
"""
from __future__ import annotations
import argparse, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def split_tex(tex: str):
    """Return (preamble, table_block, bibliography) -- the parts markdown never owns."""
    pre = tex[:tex.index(r"\begin{abstract}")]
    # keyed by \label so a paper can carry several; the markdown references
    # them as {{TABLE:tab:judge}} and never contains their digits
    table = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
        table[lab.group(1) if lab else f"_{len(table)}"] = m.group(0)
    bib = tex[tex.index(r"\begin{thebibliography}"):]
    # the slice runs to EOF and so already carries \end{document}; drop it here so
    # the rebuilder can append exactly one and repeated round trips stay stable
    bib = re.sub(r"\s*\\end\{document\}\s*$", "\n", bib)
    return pre, table, bib


def tex_to_md(tex: str) -> str:
    pre, table, bib = split_tex(tex)
    body = tex[tex.index(r"\begin{abstract}"):tex.index(r"\begin{thebibliography}")]
    for lab, blk in table.items():
        body = body.replace(blk, "{{TABLE:" + lab + "}}")
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  lambda m: "## Abstract\n\n" + m.group(1).strip(), body, flags=re.S)
    # Keep the \label: dropping it silently breaks every \S\ref in the paper,
    # and the breakage is invisible until you read the built PDF.
    body = re.sub(r"\\section\*?\{([^}]*)\}\\label\{([^}]*)\}", r"## \1 {#\2}\n", body)
    body = re.sub(r"\\section\*?\{([^}]*)\}", r"## \1\n", body)
    body = re.sub(r"\\noindent\s*", "", body)
    body = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", body)
    body = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", body)
    body = re.sub(r"\\end\{document\}", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    title = re.search(r"\\title\{(.*?)\}\s*\n", pre, re.S)
    head = ("<!-- Edit the prose here, then: python scripts/md_to_tex.py\n"
            "     The table, preamble and bibliography come from the .tex and are\n"
            "     not editable here on purpose -- their numbers are gated. -->\n\n")
    if title:
        head += "# " + title.group(1).replace("\\\\", " ").strip() + "\n\n"
    return head + body.strip() + "\n"


def md_to_tex(md: str, tex: str) -> str:
    pre, table, bib = split_tex(tex)
    body = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    body = re.sub(r"^#[^\n#][^\n]*(?:\n(?!\s*$)[^\n]*)*", "", body, count=1, flags=re.M)  # title lives in the preamble
    out, in_abstract = [], False
    blocks = re.split(r"\n\s*\n", body.strip())
    i = -1
    while i + 1 < len(blocks):
        i += 1
        b = blocks[i].strip()
        if not b:
            continue
        if b.startswith("## "):
            head, _, rest = b.partition("\n")
            name = head[3:].strip()
            lab = re.search(r"\s*\{#([^}]*)\}\s*\Z", name)
            if lab:
                name = name[:lab.start()].strip()
            if name.lower() == "abstract":
                out.append(r"\begin{abstract}"); in_abstract = True
            else:
                if in_abstract:
                    out.append(r"\end{abstract}"); in_abstract = False
                out.append(r"\section{" + name + "}"
                           + (r"\label{" + lab.group(1) + "}" if lab else ""))
            if rest.strip():
                blocks.insert(i + 1, rest.strip())   # process the body as its own block
            continue
        # A marker can be glued to the paragraph after it, because in the .tex
        # the prose sometimes resumes on the same line as \end{table}. Split it
        # off and let the remainder be processed as its own block, exactly as
        # headings do above -- otherwise the marker silently survives into the
        # .tex as literal text and the rebuild drops the table.
        mlead = re.match(r"(\{\{TABLE:[^}]*\}\})\s*(.+)\Z", b, re.S)
        if mlead:
            b = mlead.group(1)
            blocks.insert(i + 1, mlead.group(2).strip())
        mt = re.fullmatch(r"\{\{TABLE:([^}]*)\}\}", b)
        if mt:
            if in_abstract:
                out.append(r"\end{abstract}"); in_abstract = False
            if mt.group(1) not in table:
                sys.exit(f"unknown table marker {{{{TABLE:{mt.group(1)}}}}} -- "
                         f"known: {sorted(table)}")
            out.append(table[mt.group(1)])
            continue
        t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", b)
        t = re.sub(r"(?<![*\\])\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", t)
        # Em-dash removal is for prose only. Inside a tabular, "---" is an empty
        # cell and rewriting it to "," silently corrupts the table.
        if r"\begin{tabular}" not in t:
            t = t.replace(" --- ", ", ").replace("---", ", ")
        out.append(t)
    if in_abstract:
        out.append(r"\end{abstract}")
    tex_out = pre + "\n\n".join(out) + "\n\n" + bib.rstrip() + "\n" + r"\end{document}" + "\n"
    return re.sub(r"\n{3,}", "\n\n", tex_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default="writeup_interpscience.tex")
    ap.add_argument("--md", default=None,
                    help="prose file; defaults to the .tex name with .txt (or .md if that exists)")
    ap.add_argument("--extract", action="store_true",
                    help="generate the .md FROM the .tex (start here)")
    a = ap.parse_args()
    tex_p = os.path.join(ROOT, a.tex)
    if a.md:
        md_p = os.path.join(ROOT, a.md)
    else:
        # .txt is the default so the prose opens in a plain editor rather than a
        # markdown previewer. .md is still honoured if that is what exists, but
        # never both: two editable copies of one paper is how files drift apart.
        txt = os.path.join(ROOT, a.tex.replace(".tex", ".txt"))
        md = os.path.join(ROOT, a.tex.replace(".tex", ".md"))
        if os.path.exists(txt) and os.path.exists(md):
            sys.exit(f"both {os.path.basename(txt)} and {os.path.basename(md)} exist; "
                     "delete one so there is a single source of truth")
        md_p = md if (os.path.exists(md) and not os.path.exists(txt)) else txt
    tex = open(tex_p).read()

    if a.extract:
        if os.path.exists(md_p):
            sys.exit(f"refusing to overwrite {os.path.basename(md_p)} -- delete it first")
        open(md_p, "w").write(tex_to_md(tex))
        print(f"wrote {os.path.basename(md_p)} -- edit that, then rerun without --extract")
        return

    if not os.path.exists(md_p):
        sys.exit(f"{os.path.basename(md_p)} not found; run with --extract first")
    md_src = open(md_p).read()
    new = md_to_tex(md_src, tex)
    # the table must survive byte-for-byte, or the gated numbers moved
    # A table may legitimately be dropped by deleting its marker. What must never
    # happen is a table surviving in ALTERED form, which would mean its gated
    # numbers were edited. So: referenced tables must appear byte-identical;
    # unreferenced ones are reported as deliberate removals.
    _, tables_before, _ = split_tex(tex)
    for lab, blk in tables_before.items():
        referenced = ("{{TABLE:" + lab + "}}") in md_src
        if referenced and blk not in new:
            sys.exit(f"REFUSING TO WRITE: verified table {lab} was altered by the rebuild")
        if not referenced:
            print(f"note: table {lab} is no longer referenced and has been dropped")
    open(tex_p, "w").write(new)
    print(f"wrote {os.path.basename(tex_p)} from {os.path.basename(md_p)}")
    print("now run: bash scripts/check_paper.sh")


if __name__ == "__main__":
    main()
