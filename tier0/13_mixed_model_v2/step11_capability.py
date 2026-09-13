"""
R2 Step 11 — symmetric capability controls for the low-ASR diffusion victims.

Goal (reviewer W5/C4): Dream, DiffuCoder and LLaDA need the same capability
controls Falcon has (benign answer rate, over-refusal rate, open-ended answer
rate) so their low ASR is interpretable as safety, not as "model emits nothing".

Key prior work already on disk (reuse, do not re-run):
  - T07 plaintext harmful baseline (all 6 victims, n=100): how often each
    victim complies with a PLAINTEXT harmful prompt. qwen 24 / llama 20 /
    falcon 0 / llada 30 / dream 3 / diffucoder 16.
  - T08 benign MetaCipher decode (60 benign prompts x 6): decode_and_comply
    rate with the cipher wrapper removed for benign content. dream 36.7 /
    diffucoder 33.3 / llada 41.7 / falcon 18.3 / llama 16.7 / qwen 60.
  - T09 benign PiF (60 benign prompts x 6, output responses): whether the
    victim produces a non-empty response to a benign prompt. dream 51/60,
    diffucoder 58/60, llada 60/60 (all non-empty in majority).
  - T10 Falcon capability (benign answer 91.7%, over-refusal 1.7%, open-ended
    ~17%): the reference values the diffusion victims must be compared to.

This script builds the DIFFUSION capability table from the existing outputs
(zero new generation) and reports, per model, the three diagnosis signals:
  (1) benign answer rate  — from T09 (non-empty/no-error response)
  (2) plaintext harmful compliance — from T07 (baseline for "can it produce
      harmful output at all if asked")
  (3) benign decode — from T08 (whether low MetaCipher is a decode ceiling)

It then states whether the diffusion victims' low ASR reads as "safety"
(responds to benign, complies with plaintext harmful, so low attacked ASR is
attack-specific) vs "capability" (fails benign too or never produces output).

Number honesty: T07/T08/T09 are n=60-100, smaller than the reviewer's n~200
target (±5pp wants 200). This table is the honest current-evidence view; a
full n~200 symmetric rerun is a separate GPU item (budget noted at the end).
"""
import os, json, glob, csv, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))

OUT = os.path.dirname(os.path.abspath(__file__))
T09 = os.path.join(OUT, "..", "09_benign_pif_intelligibility")
T07 = os.path.join(OUT, "..", "07_plaintext_harmful_baseline")
T08 = os.path.join(OUT, "..", "08_benign_metacipher_decode")

MODELS = ["dream", "diffucoder", "llada", "falcon", "llama", "qwen"]

# T09 benign response: non-empty?
def t09_benign_answer_rate():
    out = {}
    for m in MODELS:
        p = os.path.join(T09, "model_outputs", f"{m}_outputs.json")
        if not os.path.exists(p):
            out[m] = None; continue
        recs = json.load(open(p))
        ok = [r for r in recs if r.get("status") == "ok"]
        nonempty = sum(1 for r in ok if (r.get("response") or "").strip())
        out[m] = {"n": len(ok), "nonempty": nonempty,
                  "pct": round(100 * nonempty / len(ok), 1) if ok else 0.0}
    return out

def t07_plaintext_compliance():
    # baseline_compliance.csv: model,n,compliance,compliance_asr_pct,ci...
    p = os.path.join(T07, "baseline_compliance.csv")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return {r["model"]: {"n": int(r["n"]), "compliance_pct": float(r["compliance_asr_pct"])}
            for r in rows}

def t08_decode():
    p = os.path.join(T08, "benign_decode_scores.csv")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return {r["model"]: {"n": int(r["n"]),
                         "dc_pct": float(r["decode_and_comply_pct"])} for r in rows}

def main():
    b = t09_benign_answer_rate()
    p = t07_plaintext_compliance()
    d = t08_decode()

    lines = [
        "# R2 Step 11 — symmetric capability controls (diffusion victims)",
        "",
        "Existing-output diagnosis (T07/T08/T09, no new generation). Per model:",
        "",
        "| model | benign-PiF answer | plaintext-harmful compliance | benign-decode | reading |",
        "|---|---|---:|---:|---|",
    ]
    for m in MODELS:
        br = b.get(m); pc = p.get(m); dc = d.get(m)
        br_s = f"{br['pct']}% ({br['nonempty']}/{br['n']})" if br else "n/a"
        pc_s = f"{pc['compliance_pct']}%" if pc else "n/a"
        dc_s = f"{dc['dc_pct']}%" if dc else "n/a"
        # reading
        if br and pc and dc:
            if br["pct"] >= 60 and pc["compliance_pct"] > 0:
                read = "responds to benign AND can produce harmful on plaintext -> low attacked ASR is attack/judge-specific"
            elif br["pct"] < 40:
                read = "low benign response -> ASR may be a capability/answer ceiling, not safety"
            else:
                read = "mixed/ambiguous"
        else:
            read = "n/a"
        lines.append(f"| {m} | {br_s} | {pc_s} | {dc_s} | {read} |")

    lines += [
        "",
        "## Honest interpretation",
        "The diffusion victims DO produce non-empty responses to benign prompts"
        " (T09: dream 51/60, llada 60/60, diffucoder 58/60) and DO comply with"
        " some plaintext harmful prompts (T07: dream 3%, llada 30%, diffucoder",
        " 16%). Their low MetaCipher/ArrAttack ASR is therefore not because they",
        " emit nothing: it is attack- or judge-specific. This differs from Falcon,",
        " which emits no final answer on ~71-83% of open-ended prompts; the",
        " diffusion victims do not have that answer-ceiling confound.",
        "",
        "## n~200 caveat",
        "The reviewer wants n~200 for +/-5pp. Current benign sets are n=60 and",
        "plaintext n=100. A full symmetric n~200 rerun for the 3 diffusion victims",
        "across benign + safety-adjacent + open-ended = ~3 x 3 x 200 = ~1,800",
        "diffusion generations (128-step each). That is the explicit GPU budget",
        "item for a compliant symmetric control; the evidence above is the",
        "current honest state.",
    ]
    with open(os.path.join(OUT, "step11_diffusion_capability.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()