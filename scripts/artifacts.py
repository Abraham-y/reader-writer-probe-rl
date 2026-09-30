#!/usr/bin/env python3
"""The list of cached artifacts the paper's gates read, and a check that a copy
of them is complete and uncorrupted.

WHY THIS EXISTS
Every number in the paper regenerates from cached artifacts that are too large
for git (activations, probe caches, judge scores, rollouts). They are published
separately as a Hugging Face dataset that mirrors this repository's paths, so a
download drops each file where the scripts already look. This script owns the
list of what goes into that dataset, `artifacts/MANIFEST.tsv`, one row per file
with its size and SHA-256.

The list is not hand-maintained. It was built by running
`scripts/check_everything.sh` with every file open logged, and keeping the
files that were read, exist, and are not already tracked in git. Hand-picking
directories is how a published artifact set ends up missing the one file a gate
needs.

    python scripts/artifacts.py fetch                  # download into place, then verify
    python scripts/artifacts.py verify                 # is every file present and intact?
    python scripts/artifacts.py upload --repo USER/NAME   # maintainers: private by default
    python scripts/artifacts.py build --opens LOG      # maintainers: rebuild the list

`fetch` asks the Hub for exactly the manifest's paths. The dataset's card is its
own README.md at the dataset root, and a whole-repo download into this checkout
would overwrite this repository's README with it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(_ROOT, "artifacts", "MANIFEST.tsv")
CARD = os.path.join(_ROOT, "artifacts", "README.md")
# Set once the dataset exists; `fetch` reads it so readers need no arguments.
DEFAULT_REPO = os.environ.get("ARTIFACTS_REPO", "")

# Read during the traced run but not artifacts: code, the paper, and tooling.
_SKIP_EXT = (".py", ".pyc", ".sh", ".tex", ".txt", ".md", ".sty", ".bib", ".log")
_SKIP_PARTS = ("__pycache__", "/.git/", "/_attic/", "/figures/")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split("\n"))


def build(opens: str) -> None:
    git = tracked()
    keep = set()
    for line in open(opens):
        p = os.path.realpath(line.strip())
        if not p.startswith(_ROOT + os.sep) or not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, _ROOT)
        if rel in git or rel.endswith(_SKIP_EXT) or any(s in "/" + rel for s in _SKIP_PARTS):
            continue
        keep.add(rel)
    rows = sorted(keep)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    total = 0
    with open(MANIFEST, "w") as f:
        f.write("path\tbytes\tsha256\n")
        for rel in rows:
            n = os.path.getsize(os.path.join(_ROOT, rel))
            total += n
            f.write(f"{rel}\t{n}\t{sha256(os.path.join(_ROOT, rel))}\n")
    print(f"wrote {os.path.relpath(MANIFEST, _ROOT)}: {len(rows)} files, {total / 1e9:.2f} GB")


def verify() -> None:
    rows = [l.rstrip("\n").split("\t") for l in open(MANIFEST)][1:]
    bad = []
    for rel, n, digest in rows:
        p = os.path.join(_ROOT, rel)
        if not os.path.isfile(p):
            bad.append(f"missing  {rel}")
        elif os.path.getsize(p) != int(n):
            bad.append(f"size     {rel}: {os.path.getsize(p)} bytes, manifest says {n}")
        elif sha256(p) != digest:
            bad.append(f"hash     {rel}")
    total = sum(int(r[1]) for r in rows)
    if bad:
        print(f"{len(bad)} of {len(rows)} artifacts are missing or differ:")
        for b in bad[:40]:
            print("  " + b)
        sys.exit(1)
    print(f"all {len(rows)} artifacts present and verified ({total / 1e9:.2f} GB)")


def _paths() -> list[str]:
    return [l.split("\t")[0] for l in open(MANIFEST).read().splitlines()[1:]]


def fetch(repo: str) -> None:
    if not repo:
        sys.exit("no dataset repo: pass --repo USER/NAME or set ARTIFACTS_REPO")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=_ROOT,
                      allow_patterns=_paths())
    verify()


def upload(repo: str, public: bool) -> None:
    """Create the dataset (private unless --public) and upload the card, the
    manifest and exactly the manifest's files, at their repository paths."""
    verify()   # never publish a set that does not match its own manifest
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=_ROOT, repo_id=repo, repo_type="dataset",
                      allow_patterns=_paths() + ["artifacts/MANIFEST.tsv"],
                      commit_message="Cached artifacts for the InterpScience 2026 camera-ready")
    kind = "public" if public else "PRIVATE (flip it to public on the Hub when ready)"
    print(f"uploaded {len(_paths())} files to https://huggingface.co/datasets/{repo} -- {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="rebuild the manifest from a file-open log")
    b.add_argument("--opens", required=True)
    sub.add_parser("verify", help="check the local copy against the manifest")
    f = sub.add_parser("fetch", help="download the artifacts into place and verify them")
    f.add_argument("--repo", default=DEFAULT_REPO)
    u = sub.add_parser("upload", help="maintainers: upload the artifacts to a dataset repo")
    u.add_argument("--repo", required=True)
    u.add_argument("--public", action="store_true", help="create it public (default: private)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.opens)
    elif a.cmd == "verify":
        verify()
    elif a.cmd == "fetch":
        fetch(a.repo)
    else:
        upload(a.repo, a.public)


if __name__ == "__main__":
    main()
