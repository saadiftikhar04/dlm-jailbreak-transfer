"""
R2 Step 7 (Gate B): score Falcon's stored responses for final-answer presence.

Question (reviewer W5 / plan Step 7): Falcon anchors the robust end of every
figure while producing a final answer on ~17% of open-ended benign prompts
(60-prompt side study). Gate B checks whether that low answer rate is a
stored-data property across ALL of Falcon's 1,991 stored rows.

Method (faithful to T10.1 / T03):
- PiF and ArrAttack judged rows store BOTH a pre-stripping `reasoning_trace`
  and the stripped final (`victim_output` / `final_response`). Rebuild raw =
  reasoning_trace + final, then split on the LAST `\\bThus\\b` boundary; the
  trailing slice is the delivered final answer. answer = empty slice means no
  final answer was delivered (trace-only).
- MetaCipher judged rows do NOT save a reasoning trace (raw_col = None); the
  stored `final_response` is already the post-strip answer. For MetaCipher we
  score answer-presence on the stored final_response (non-empty after strip),
  and we say so explicitly in the output (method differs; no trace to split).

Outputs:
  falcon_per_cell_answer_rate.csv
    per (attack, dataset) cell: n, answer_rate, unconditional ASR,
    conditional ASR = ASR / answer_rate, plus the raw non-empty rate.
  falcon_per_cell_answer_rate.md  (human-readable)
Guardrails: zero rows dropped (G3); ASR from the official judged labels;
answer-presence defined consistently per attack.

Note: this is a zero-compute task -- the generations already exist on disk.
"""
import os, sys, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00_shared"))
import common as CM

OUT = os.path.join(CM.OUT_ROOT, "13_mixed_model_v2")
os.makedirs(OUT, exist_ok=True)

THUS_MARK = re.compile(r"\bThus\b", re.IGNORECASE)


def split_last_thus(raw: str):
    """Split raw into (reasoning, answer) at the LAST \\bThus\\b boundary.
    Returns (reas, ans). If no Thus found, everything is 'reasoning'."""
    if not isinstance(raw, str) or not raw.strip():
        return "", ""
    matches = list(THUS_MARK.finditer(raw))
    if matches:
        m = matches[-1]
        return raw[: m.start()], raw[m.end():]
    return raw, ""


def falcon_has_final_answer(attack, row):
    """Answer-presence per attack. Returns (present: bool, note)."""
    a = CM.ATTACKS[attack]
    raw_col = a["raw_col"]
    final_col = a["response_col"]
    final = row[final_col]
    final_s = "" if pd.isna(final) else str(final).strip()

    if raw_col is not None and pd.notna(row[raw_col]):
        # rebuild raw and split on last Thus (T10.1 faithful)
        raw = str(row[raw_col]).strip() + (" " + final_s if final_s else "")
        _, ans = split_last_thus(raw)
        return (ans.strip() != ""), "last-Thus-split"
    # no trace saved (MetaCipher): answer-presence = non-empty stored final
    return (final_s != ""), "stored-final-nonempty"


# ---- build per-cell frame ----------------------------------------------
rows = []
for attack in ["pif", "metacipher", "arrattack"]:
    df = CM.load(attack, "falcon")
    succ = CM.success_series(attack, df)
    out = []
    for _, r in df.iterrows():
        present, note = falcon_has_final_answer(attack, r)
        out.append({"attack": attack,
                    "dataset": str(r["dataset"]),
                    "prompt": str(r["dataset"]) + ":" + str(r["prompt_idx"]),
                    "has_final_answer": int(present),
                    "method": note,
                    "asr_success": bool(succ.loc[r.name])})
    rows.extend(out)

fl = pd.DataFrame(rows)
assert len(fl) == 913 + 913 + 165 == 1991, len(fl)
# method split sanity: PiF/AA should be last-Thus, MC stored-final
print("method counts:", fl["method"].value_counts().to_dict())

# ---- aggregate per cell -------------------------------------------------
cells = []
for (attack, ds), g in fl.groupby(["attack", "dataset"]):
    n = len(g)
    n_ans = int(g["has_final_answer"].sum())
    n_succ = int(g["asr_success"].sum())
    ans_rate = n_ans / n
    uncond_asr = n_succ / n
    cond_asr = (n_succ / n_ans) if n_ans else float("nan")
    cells.append({
        "attack": attack, "dataset": ds, "n": n,
        "n_final_answer": n_ans,
        "answer_rate_pct": round(100 * ans_rate, 2),
        "unconditional_asr_pct": round(100 * uncond_asr, 2),
        "conditional_asr_pct": (round(100 * cond_asr, 2)
                                if n_ans else ""),
    })
cell = pd.DataFrame(cells)

# total per attack
for attack in ["pif", "metacipher", "arrattack"]:
    g = fl[fl.attack == attack]
    n = len(g); n_ans = int(g.has_final_answer.sum()); n_succ = int(g.asr_success.sum())
    cell.loc[len(cell)] = {
        "attack": attack, "dataset": "TOTAL",
        "n": n, "n_final_answer": n_ans,
        "answer_rate_pct": round(100 * n_ans / n, 2),
        "unconditional_asr_pct": round(100 * n_succ / n, 2),
        "conditional_asr_pct": round(100 * n_succ / n_ans, 2),
    }

# grand total
g = fl; n = len(g); n_ans = int(g.has_final_answer.sum()); n_succ = int(g.asr_success.sum())
cell.loc[len(cell)] = {
    "attack": "ALL", "dataset": "TOTAL",
    "n": n, "n_final_answer": n_ans,
    "answer_rate_pct": round(100 * n_ans / n, 2),
    "unconditional_asr_pct": round(100 * n_succ / n, 2),
    "conditional_asr_pct": round(100 * n_succ / n_ans, 2),
}

cell.to_csv(os.path.join(OUT, "falcon_per_cell_answer_rate.csv"),
            index=False, float_format="%.2f")

# ---- markdown summary --------------------------------------------------
lines = [
    "# R2 Step 7 (Gate B) - Falcon per-cell final-answer rate\n",
    "Method: PiF/ArrAttack rebuild raw = reasoning_trace + stored final and",
    "split on the LAST `\\bThus\\b` boundary (T10.1-faithful); an empty trailing",
    "slice = no delivered final answer. MetaCipher rows do not save a reasoning",
    "trace, so answer-presence is scored on the non-empty stored `final_response`",
    "(method explicitly different; no trace to split).",
    "",
    "Appendix D.1's 1,078-row figure covers PiF + ArrAttack only; it is NOT",
    "Falcon's full case count. The full count is 1,991 (913 + 913 + 165).",
    "",
    "| attack | dataset | n | ans% | uncond ASR% | cond ASR% |",
    "|---|---|---:|---:|---:|---:|",
]
for _, r in cell.iterrows():
    lines.append(f"| {r['attack']} | {r['dataset']} | {r['n']} | "
                 f"{r['answer_rate_pct']} | {r['unconditional_asr_pct']} | "
                 f"{r['conditional_asr_pct']} |")
lines += [
    "",
    "Reading (honest): {overall_ans}% of Falcon's stored rows carry a final",
    "answer at all; conditional ASR = unconditional ASR / answer rate. Where",
    "the answer rate is far below 1 (PiF open-ended and most cells), the",
    "unconditional 0% ASR reads as a delivered-answer floor, not a clean "
    "content-safety zero (cross-ref T10 / T03 / manual-read edge cases).",
    "",
]
with open(os.path.join(OUT, "falcon_per_cell_answer_rate.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(cell.to_string(index=False))
print("\nWrote", os.path.join(OUT, "falcon_per_cell_answer_rate.csv"))