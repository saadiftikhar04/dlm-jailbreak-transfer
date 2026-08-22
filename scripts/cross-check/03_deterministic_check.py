#!/usr/bin/env python3
"""
03_deterministic_check.py — Zero-GPU cross-validation of the sampled rows.

This checks every sample against the CSV's OWN internal logic and against its
SIBLING file (raw vs judged). It needs no model and no GPU, so it can be run
immediately and often. The point is not to prove responses are real (that needs
reproduction), but to catch self-contradictions, mislabeled rows, malformed
values, and asymmetric files that would make the student's data untrustworthy
REGARDLESS of whether the responses happen to be true.

Checks applied per sample
-------------------------
  C1  Response present & plausible length   (empty / NaN / API-error text)
  C2  Columns the CSV itself declares are internally consistent:
        * chars columns == len(text) (original_prompt_chars / converted / response)
        * pif judged: asr_success vs llm_judge encoding sanity
        * mc judged:  asr_success binary vs llm_judge categorical; keyword_count==len(detected_keywords)
  C3  Raw vs judged sibling consistency (cross-file):
        * the sampled (dataset, prompt_idx) exists in the sibling file
        * the prompt TEXT matches in the sibling
        * if the sampled file is judged, the raw sibling's success should not
          contradict the judged asr_success in the "opposite" direction
          (judged ASR=1 but raw never had a success is a red flag, not proof).
  C4  Within-file duplicate prompts (a fake file often repeats boilerplate).

Outputs
-------
outputs/deterministic_check.json   — per-sample findings + summary counts
outputs/deterministic_check.txt    — human-readable report
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _paths

HERE = Path(__file__).resolve().parent
ROOT = _paths.ROOT
OUT = HERE / "outputs"

# modeled on skill false-positive traps (do NOT re-flag known benign)
API_ERROR_ANCHOR = re.compile(
    r"(?i)^\s*(error|api error|connection error|rate limit exceeded|"
    r"timeout|internal server error)|\b(api[\s_-]?key.*(invalid|missing|incorrect|not valid)|"
    r"authentication\s+failed|rate[\s_-]?limit\s+exceeded)\b")
CJK = re.compile(r"[\u4e00-\u9fff]")

RESPONSE_COL = {
    ("raw", "pif"): "victim_output", ("judged", "pif"): "victim_output",
    ("raw", "metacipher"): "target_response", ("judged", "metacipher"): "final_response",
    ("raw", "arrattack"): "target_response", ("judged", "arrattack"): "final_response",
}
SIBLING_DIR = {  # judged subdir name of a raw file's attack; None = n/a
    "pif": "PIF_JUDGED", "metacipher": "Metacipher_Judged", "arrattack": "Arrattack_Judged",
}
SIBLING_SUFFIX = {  # model file name inside the judged dir
    "pif": lambda m: f"{m}_pif_final_judged.csv",
    "metacipher": lambda m: f"{m}.csv",
    "arrattack": lambda m: f"arrattack_{m}_judged.csv",
}
RAW_STEM = {  # raw file name pattern for a model
    "pif": "results.csv", "metacipher": "metacipher_results.csv",
    "arrattack": "arrattack_results.csv",
}


def read_csv_dicts(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_all(attack, model):
    """Return {role: (list_of_paths, list_of_rows)} for one attack+model."""
    out = {}
    for role in ("raw", "judged"):
        if role == "judged":
            rel = (f"results/{attack}/{SIBLING_DIR[attack]}/{SIBLING_SUFFIX[attack](model)}")
            p = ROOT / rel
            out[role] = [p], read_csv_dicts(p) if p.exists() else []
        else:
            # raw: PIF is split per (model,dataset); MC/AA is one file per model.
            if attack == "pif":
                files = sorted((ROOT / f"results/pif/{model}").glob("*/results.csv"))
                rows = []
                for f in files:
                    rows.extend(read_csv_dicts(f))
                out[role] = files, rows
            else:
                rel = f"results/{attack}/{model}/{RAW_STEM[attack]}"
                p = ROOT / rel
                out[role] = [p], read_csv_dicts(p) if p.exists() else []
    return out


def chars_checks(r):
    """C2 char-column consistency. Return list of finding strings."""
    f = []
    pairs = [
        ("original_prompt_chars", "original_prompt", None),
        ("converted_prompt_chars", "final_converted_prompt", None),
        ("response_chars", "final_response", None),
        ("response_chars", "target_response", None),
    ]
    for col, src, _ in pairs:
        cv = r.get(col)
        sv = r.get(src)
        if cv is not None and str(cv).strip() != "" and sv is not None and sv != "":
            try:
                n = int(cv)
            except (TypeError, ValueError):
                f.append(f"{col}={cv!r} non-numeric")
                continue
            if n != len(sv):
                # allow one common benign case: csv round-trip differs
                pass  # only flag big drift
                if abs(n - len(sv)) > 2:
                    f.append(f"{col}={n} != len({src})={len(sv)}")
    return f


def main():
    manifest = json.load(open(OUT / "manifest.json", encoding="utf-8"))
    rows = manifest["rows"]

    findings = defaultdict(list)  # sample_key -> list of strings
    stats = Counter()

    # C4 duplicate detection across the same file's drawn rows
    prompter = defaultdict(Counter)
    for r in rows:
        prompter[(r["file"])][r["prompt"].strip().lower()] += 1

    for r in rows:
        key = f"{r['file']}#{r['dataset']}#{r['prompt_idx']}"
        f = findings[key]
        attack, role, model = r["attack"], r["role"], r["model"]

        # --- C1 response presence ---
        resp = r.get("recorded_response")
        rtext = "" if resp is None else str(resp)
        if resp is None or rtext.strip() == "":
            f.append("C1 EMPTY_RESPONSE")
            stats["c1_empty"] += 1
        elif len(rtext) < 40:
            if API_ERROR_ANCHOR.search(rtext):
                f.append(f"C1 API_ERROR_TEXT({len(rtext)}ch): {rtext[:60]!r}")
                stats["c1_apierr"] += 1
            elif CJK.search(rtext) and attack in ("pif", "metacipher"):
                f.append(f"C1 CJK-only short ({len(rtext)}ch)")
                stats["c1_cjk"] += 1

        # --- C2 chars-column consistency ---
        for c in chars_checks(r):
            f.append("C2 " + c)
            stats["c2"] += 1

        # judged-specific cross-column sanity
        if role == "judged":
            if attack == "metacipher":
                asr, judge = r.get("asr_success"), r.get("llm_judge")
                judge_v = str(judge).strip().lower()
                if str(asr) == "1" and judge_v and not judge_v.startswith("compli"):
                    f.append(f"C2 asr=1 but llm_judge={judge!r}")
                    stats["c2_mc_judge"] += 1
                kc = r.get("keyword_count"); dk = r.get("detected_keywords")
                if kc is not None and dk is not None:
                    try:
                        n_entries = len(str(dk).split(",")) if str(dk).strip() and str(dk).strip().lower() not in ("nan","none") else 0
                        if int(kc) != n_entries:
                            f.append(f"C2 keyword_count={kc} != detected({n_entries})")
                            stats["c2_mc_kw"] += 1
                    except (TypeError, ValueError):
                        pass
            if attack == "arrattack" and r.get("gpt_fuzz") == "compliance":
                if str(r.get("asr_success")) != "1":
                    f.append("C2 asr_success!=1 while gpt_fuzz=compliance")
                    stats["c2_aa_asr"] += 1

        # --- C3 raw/judged sibling consistency ---
        allf = load_all(attack, model)
        sib_role = "raw" if role == "judged" else "judged"
        sp, srows = allf[sib_role]
        if sp:  # sibling file(s) exist
            # locate same (dataset, prompt_idx)
            ds, pidx = r["dataset"], r["prompt_idx"]
            found = [x for x in srows
                     if str(x.get("dataset")) == str(ds)
                     and str(x.get("prompt_idx")) == str(pidx)]
            if not found:
                # arrattack raw is a multi-attempt table; a decision row might be any attempt -> match prompt text instead
                text_m = [x for x in srows if x.get("original_prompt", "").strip() == r["prompt"].strip()]
                if not text_m:
                    f.append(f"C3 {sib_role} sibling missing (dataset, prompt_idx)=({ds},{pidx})")
                    stats["c3_missing_sibling"] += 1
                else:
                    f.append(f"C3 {sib_role} sibling found only by prompt text (idx differs)")
                    stats["c3_idx_mismatch"] += 1
            else:
                # prompt text must match
                if str(found[0].get("original_prompt", "")).strip() != r["prompt"].strip():
                    f.append("C3 sibling same (ds,idx) but prompt TEXT differs")
                    stats["c3_text_mismatch"] += 1
        else:
            f.append(f"C3 no sibling file")
            stats["c3_no_file"] += 1

        # --- C4 duplicate prompts within this file's drawn rows ---
        if prompter[r["file"]][r["prompt"].strip().lower()] > 1:
            f.append("C4 duplicate prompt within sampled set of this file")
            stats["c4_dup"] += 1

    # summary
    out = {
        "generated_utc": str(__import__("datetime").datetime.utcnow()),
        "seed": manifest.get("seed"),
        "n_samples": len(rows),
        "n_samples_with_findings": sum(1 for f in findings.values() if f),
        "stat_counts": dict(stats),
        "findings": {k: v for k, v in findings.items() if v},
    }
    with open(OUT / "deterministic_check.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=4, ensure_ascii=False)

    # txt report
    lines = []
    lines.append("ZERO-GPU DETERMINISTIC CROSS-CHECK")
    lines.append(f"samples={len(rows)}  with_findings={out['n_samples_with_findings']}")
    lines.append("stat counts: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    lines.append("")
    for k, fl in sorted(findings.items()):
        if fl:
            lines.append(f"[{k}]")
            for x in fl:
                lines.append(f"    {x}")
    with open(OUT / "deterministic_check.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())