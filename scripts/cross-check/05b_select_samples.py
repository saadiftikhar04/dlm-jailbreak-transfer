#!/usr/bin/env python3
"""
05b_select_samples.py — Filter the manifest to the rows requested for reproduction.
Prints TSV: attack<TAB>role<TAB>model<TAB>dataset<TAB>prompt_idx<TAB>prompt
Consumed by 05_reproduce.sh.
"""
import json
import sys
from pathlib import Path

import _paths

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    attack = sys.argv[1] if len(sys.argv) > 1 else ""
    model = sys.argv[2] if len(sys.argv) > 2 else ""
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    rows = json.load(open(OUT / "manifest.json", encoding="utf-8"))["rows"]
    sel = []
    for r in rows:
        if attack and r["attack"] != attack:
            continue
        if model and ((r["model"] != model) and (r["model"] != model)):
            continue
        sel.append(r)
    # prefer judged samples first (they carry asr_success), then raw
    sel.sort(key=lambda r: (0 if r["role"] == "judged" else 1))
    if limit:
        sel = sel[:limit]
    for r in sel:
        # tab-safe prompt (sanitise any tabs)
        p = str(r["prompt"]).replace("\t", " ").replace("\n", " ").strip()
        out = "\t".join([r["attack"], r["role"], r["model"], str(r["dataset"]),
                         str(r["prompt_idx"]), p])
        print(out)


if __name__ == "__main__":
    main()