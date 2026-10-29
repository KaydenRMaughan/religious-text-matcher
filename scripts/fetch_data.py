"""Download the public demo corpus and evaluation labels into data/raw/.

Sources (see README for licenses):
- King James Bible, Project Gutenberg #10 (public domain, US)
- Book of Mormon, Project Gutenberg #17 (public domain, US)
- OpenBible.info cross-references (CC BY)

Files already on disk are skipped unless --force is given. Each download is
recorded in data/raw/SOURCES.json with its URL, retrieval date and SHA-256.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
USER_AGENT = "religious-text-matcher/0.1 (+https://github.com/KaydenRMaughan/religious-text-matcher)"

SOURCES = {
    "kjv.txt": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "book_of_mormon.txt": "https://www.gutenberg.org/cache/epub/17/pg17.txt",
    "cross_references.zip": "https://a.openbible.info/data/cross-references.zip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "SOURCES.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for i, (name, url) in enumerate(SOURCES.items()):
        dest = RAW_DIR / name
        if dest.exists() and not args.force:
            print(f"skip   {name} (exists)")
        else:
            if i:
                time.sleep(1)  # be polite to the hosts
            print(f"fetch  {name} <- {url}")
            download(url, dest)
            manifest[name] = {
                "url": url,
                "retrieved": dt.date.today().isoformat(),
                "sha256": sha256(dest),
            }

    xref_zip = RAW_DIR / "cross_references.zip"
    with zipfile.ZipFile(xref_zip) as z:
        members = [m for m in z.namelist() if m.endswith(".txt")]
        if len(members) != 1:
            print(f"error: expected one .txt in {xref_zip.name}, found {members}", file=sys.stderr)
            return 1
        (RAW_DIR / "cross_references.txt").write_bytes(z.read(members[0]))

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done   {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
