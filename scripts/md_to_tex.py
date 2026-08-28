"""Edit the paper as markdown; rebuild the LaTeX from it.

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

    python scripts/md_to_tex.py --extract       # .tex -> .md  (once, to start)
    python scripts/md_to_tex.py                 # .md  -> .tex (after editing)

MARKDOWN YOU CAN USE
    ## Heading            -> \\section{Heading}
    **bold**              -> \\textbf{bold}
    *italic*              -> \\emph{italic}
    {{TABLE}}             -> the verified table, spliced from the .tex
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
    m = re.search(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S)
    table = m.group(0) if m else ""
    bib = tex[tex.index(r"\begin{thebibliography}"):]
    # the slice runs to EOF and so already carries \end{document}; drop it here so
    # the rebuilder can append exactly one and repeated round trips stay stable
    bib = re.sub(r"\s*\\end\{document\}\s*$", "\n", bib)
    return pre, table, bib


def tex_to_md(tex: str) -> str:
    pre, table, bib = split_tex(tex)
    body = tex[tex.index(r"\begin{abstract}"):tex.index(r"\begin{thebibliography}")]
    if table:
        body = body.replace(table, "{{TABLE}}")
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  lambda m: "## Abstract\n\n" + m.group(1).strip(), body, flags=re.S)
    body = re.sub(r"\\section\*?\{([^}]*)\}(?:\\label\{[^}]*\})?", r"## \1\n", body)
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
            if name.lower() == "abstract":
                out.append(r"\begin{abstract}"); in_abstract = True
            else:
                if in_abstract:
                    out.append(r"\end{abstract}"); in_abstract = False
                out.append(r"\section{" + name + "}")
            if rest.strip():
                blocks.insert(i + 1, rest.strip())   # process the body as its own block
            continue
        if b == "{{TABLE}}":
            if in_abstract:
                out.append(r"\end{abstract}"); in_abstract = False
            out.append(table)
            continue
        t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", b)
        t = re.sub(r"(?<![*\\])\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", t)
        t = t.replace(" --- ", ", ").replace("---", ", ")
        out.append(t)
    if in_abstract:
        out.append(r"\end{abstract}")
    tex_out = pre + "\n\n".join(out) + "\n\n" + bib.rstrip() + "\n" + r"\end{document}" + "\n"
    return re.sub(r"\n{3,}", "\n\n", tex_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default="writeup_judge_spotlight.tex")
    ap.add_argument("--md", default=None, help="defaults to the .tex name with .md")
    ap.add_argument("--extract", action="store_true",
                    help="generate the .md FROM the .tex (start here)")
    a = ap.parse_args()
    tex_p = os.path.join(ROOT, a.tex)
    md_p = os.path.join(ROOT, a.md or a.tex.replace(".tex", ".md"))
    tex = open(tex_p).read()

    if a.extract:
        if os.path.exists(md_p):
            sys.exit(f"refusing to overwrite {os.path.basename(md_p)} -- delete it first")
        open(md_p, "w").write(tex_to_md(tex))
        print(f"wrote {os.path.basename(md_p)} -- edit that, then rerun without --extract")
        return

    if not os.path.exists(md_p):
        sys.exit(f"{os.path.basename(md_p)} not found; run with --extract first")
    new = md_to_tex(open(md_p).read(), tex)
    # the table must survive byte-for-byte, or the gated numbers moved
    _, table_before, _ = split_tex(tex)
    if table_before and table_before not in new:
        sys.exit("REFUSING TO WRITE: the verified table did not survive the rebuild")
    open(tex_p, "w").write(new)
    print(f"wrote {os.path.basename(tex_p)} from {os.path.basename(md_p)}")
    print("now run: bash scripts/check_paper.sh")


if __name__ == "__main__":
    main()
