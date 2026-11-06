"""Convert the private conference pickle into the corpus format (docs/corpus-format.md).

    python scripts/build_private_corpus.py

Reads paths from config.local.toml ([private] conference_pickle) and writes
[corpus] path as JSON Lines, adding each talk's footnoted scripture citations as
``citations``. Private: the input and output live outside version control.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

from rtm.corpus import REPO_ROOT, load_local_config


def _list(value) -> list[str]:
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value) if value is not None else []


def main() -> None:
    config = load_local_config()
    if not config or "private" not in config:
        raise SystemExit("config.local.toml with a [private] section is required")
    src = Path(config["private"]["conference_pickle"])
    src = src if src.is_absolute() else REPO_ROOT / src
    out = Path(config["corpus"]["path"])

    talks = pd.read_pickle(src)
    with out.open("w", encoding="utf-8") as f:
        for t in talks.itertuples():
            slug = re.sub(r"[^a-z0-9-]", "", str(t.url).rstrip("/").rsplit("/", 1)[-1].lower())
            record = {
                "id": f"{int(t.year)}-{int(t.month):02d}-{slug}",
                "title": str(t.title).strip(), "text": str(t.body), "source": "General Conference",
                "year": int(t.year), "author": str(t.author), "url": str(t.url),
                "citations": _list(t.scripture_references),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(talks)} talks to {out}")


if __name__ == "__main__":
    main()
