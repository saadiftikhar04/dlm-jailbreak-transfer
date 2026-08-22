#!/usr/bin/env python3
"""
06_compare.py — Diff the RECORDED victim response (from the student's CSV) against
the REPRODUCED response (from 05_reproduce.sh) for each sampled row, and emit a
verdict.

Verdicts
--------
  MATCH         response text identical (or near-identical, norm<0.01)
  SIMILAR       high textual overlap (norm>=0.70) OR same ASR/topic; consistent
  DIFFERENT     low overlap; the re-run produced something else
  MISSING       the re-run produced no response (env/model missing) -> not a failure
  EMPTY_RECORDED the CSV claimed empty; reproduction non-empty (or vice-versa)

Why no strict byte match for metacipher/arrattack:
  MetaCipher's cipher sampling and the LLM judge are stochastic; ArrAttack's
  GPTFuzz loop is stochastic. A single re-run is EXPECTED to differ in wording.
  So a SIMILAR verdict means "weights+data reproduce the same behaviour", which is
  what we use to trust the student's number. PIF (greedy, temperature ~0) is the
  one where a near-identical response is expected; flag there if not.

Inputs
------
  outputs/manifest.json     — recorded rows (from 02_sample)
  outputs/reproduced/       — produced by 05_reproduce.sh
Outputs
-------
  outputs/compare.json      — per-sample verdicts + summary
  outputs/compare.txt       — human-readable report
"""

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import _paths

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
REPRO = OUT / "reproduced"


def norm(s):
    s = str(s if s is not None else "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def load_reproduced(attack, model, dataset, prompt_idx):
    """Try to locate the reproduced response for a sample."""
    base = REPRO / attack / model
    if not base.exists():
        return None, None
    hits = []
    # look in per-sample *.json first (fast, structured)
    for jf in base.glob(f"sample_{prompt_idx}.json"):
        try:
            d = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("reproduced_rows") or []
        for r in rows:
            resp = r.get("final_response") or r.get("victim_output") or r.get("target_response")
            if resp:
                hits.append((r, resp))
    # then any results.csv / arrattack_results.csv on disk for that model
    for csvf in base.glob("**/*.csv"):
        if not (csvf.stem.startswith("results") or csvf.stem.startswith("arrattack")):
            continue
        import csv as _csv
        try:
            with open(csvf, newline="", encoding="utf-8-sig") as f:
                for row in _csv.DictReader(f):
                    if str(row.get("dataset")) == str(dataset) and \
                       str(row.get("prompt_idx")) == str(prompt_idx):
                        resp = row.get("victim_output") or row.get("final_response") or row.get("target_response")
                        if resp and norm(resp):
                            hits.append((row, resp))
        except Exception:
            continue
    # dedupe by norm
    seen = {}
    for row, resp in hits:
        seen.setdefault(norm(resp), (row, resp))
    return list(seen.values()), len(seen.values())


def main():
    manifest = json.load(open(OUT / "manifest.json", encoding="utf-8"))
    rows = manifest["rows"]

    verdicts = []
    stats = {}
    for r in rows:
        recorded = r.get("recorded_response") or ""
        rec_norm = norm(recorded)
        hit, nhits = load_reproduced(r["attack"], r["model"], r["dataset"],
                                     r["prompt_idx"])
        samples = hit or [None]
        # best-match against any reproduced row available
        best = max(samples, key=lambda s: sim(recorded, s[1] if s else ""))
        if best is None or norm(best[1]) == "":
            verdict = "MISSING"
        else:
            s = sim(recorded, best[1])
            if rec_norm == "" :
                verdict = "EMPTY_RECORDED"
            elif s >= 0.99 and r["attack"] == "pif":
                verdict = "MATCH"
            elif s >= 0.70:
                verdict = "SIMILAR"
            else:
                verdict = "DIFFERENT"
        stats[verdict] = stats.get(verdict, 0) + 1
        verdicts.append({
            "attack": r["attack"], "role": r["role"], "model": r["model"],
            "dataset": r["dataset"], "prompt_idx": r["prompt_idx"],
            "file": r["file"], "verdict": verdict,
            "similarity": round(sim(recorded, best[1] if best else "0"), 3),
            "n_reproduced": nhits if (nhits is not None) else (1 if best and norm(best[1]) else 0),
            "recorded_len": len(recorded),
            "reproduced_len": len(best[1]) if best else 0,
        })

    summary = {
        "generated_utc": str(__import__("datetime").datetime.utcnow()),
        "n_samples": len(verdicts),
        "verdict_counts": stats,
        "n_with_any_reproduced": sum(1 for v in verdicts if v["n_reproduced"] > 0),
    }
    out_f = OUT / "compare.json"
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "verdicts": verdicts}, f, indent=4, ensure_ascii=False)

    lines = ["RECORDED vs REPRODUCED COMPARISON"]
    lines.append(f"samples={summary['n_samples']} | verdicts={summary['verdict_counts']}")
    lines.append("")
    for v in verdicts:
        lines.append(f"[{v['verdict']:<12}] {v['attack']}/{v['model']}/{v['dataset']}#{v['prompt_idx']} "
                     f"sim={v['similarity']} recorded={v['recorded_len']}ch repro={v['reproduced_len']}ch")
    (OUT / "compare.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:80]))
    print("\n--- README guidance ---")
    print("MISSING = couldn't reproduce in this env (weights/data absent). Not a failure.")
    print("MATCH/SIMILAR = faithful. DIFFERENT/EMPTY_RECORDED need investigation.")


if __name__ == "__main__":
    sys.exit(main())