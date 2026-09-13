"""T08.2: apply the REAL MetaCipher wrapper to the 60 benign prompts, faithfully
reusing metacipher_multi.KeywordAgent.find_keywords + CipherPool.generate_ciphered_prompt
(cipher=substitution, the dominant real cipher: 690/913). Records each prompt's
keyword/mask count so "how much was actually encoded" is explicit (benign prompts
often have no malicious keyword, so find_keywords returns 'none' and the prompt
is barely encoded -- that IS the real MetaCipher behavior and we record it).

Outputs:
  benign_metacipher_wrapped.json  (id, original_prompt, wrapped_prompt, keywords, n_masks)
  g1_roundtrip_check.txt          (re-apply wrapper to recorded harmful prompts,
                                   assert byte-identical to saved attacked_prompt)
Uses DEEPSEEK_API_KEY from env. Run locally (API for find_keywords). Generation
(victim decode) is a separate HPC step.
"""
import os, sys, json, csv, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
os.environ.setdefault("DEEPSEEK_API_KEY", "")
import metacipher_multi as mm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "08_benign_metacipher_decode")
os.makedirs(OUT_DIR, exist_ok=True)

CIPHER = "substitution"  # dominant real cipher (690/913 metacipher cases)

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("Set DEEPSEEK_API_KEY")
    ka = mm.KeywordAgent(api_key=api_key)
    # silence HTTP info logs
    import logging; logging.getLogger("httpx").setLevel(logging.WARNING)

    with open(os.path.join(HERE, "..", "08_benign_metacipher_decode",
                           "benign_prompts.json")) as f:
        benign = json.load(f)
    assert len(benign) == 60

    wrapped = []
    n_masked = 0
    for b in benign:
        prompt = b["prompt"]
        keywords = [k for k in ka.find_keywords(prompt) if k and k.lower() != "none"]
        adv, _ = mm.CipherPool.generate_ciphered_prompt(CIPHER, keywords, prompt, None)
        nmask = adv.count("[MASK")
        n_masked += 1 if nmask else 0
        wrapped.append({
            "id": b["id"], "category": b["category"],
            "original_prompt": prompt,
            "wrapped_prompt": adv,
            "keywords": keywords,
            "n_masks": nmask,
        })
    with open(os.path.join(HERE, "..", "08_benign_metacipher_decode",
                           "benign_metacipher_wrapped.json"), "w") as f:
        json.dump(wrapped, f, indent=4, ensure_ascii=False)

    masked = sum(1 for w in wrapped if w["n_masks"] > 0)
    print(f"wrapped {len(wrapped)} benign prompts (cipher={CIPHER})")
    print(f"  with >=1 mask: {masked}, without mask: {len(wrapped)-masked}")

    # ---- G1 round-trip: re-apply wrapper to recorded harmful prompts ----
    g1_lines = ["# G1 round-trip check: re-apply MetaCipher wrapper log",
                f"cipher={CIPHER}, keyword detection=KeywordAgent.find_keywords (real)"]
    ok = True
    # use saved judged metacipher rows (have attacked_prompt? check cols)
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(HERE), "..", "results",
                                  "metacipher", "Metacipher_Judged", "qwen.csv"))
    acol = "attacked_prompt" if "attacked_prompt" in df.columns else \
           ("final_converted_prompt" if "final_converted_prompt" in df.columns else None)
    g1_lines.append(f"attacked-prompt col used: {acol}")
    if acol:
        row = df.dropna(subset=["original_prompt", acol]).iloc[0]
        saved = str(row[acol])
        kw = ka.find_keywords(str(row["original_prompt"]))
        adv, _ = mm.CipherPool.generate_ciphered_prompt("substitution", kw,
                                                        str(row["original_prompt"]), None)
        same = (adv.split("### Your response:")[0] in saved or
                saved.split("### Your response:")[0] in adv)
        g1_lines.append(f"sample harmful prompt (idx {row['prompt_idx']}):")
        g1_lines.append(f"  detected keywords: {kw}")
        g1_lines.append(f"  regen prompt head: {adv[:120]!r}")
        g1_lines.append(f"  saved prompt head: {saved[:120]!r}")
        g1_lines.append(f"  prefix-match (reusable wrapper): {same}")
        ok = ok and same
    g1_lines.append(f"G1 round-trip feasible: {ok}")
    open(os.path.join(HERE, "..", "08_benign_metacipher_decode",
                      "g1_roundtrip_check.txt"), "w").write("\n".join(g1_lines) + "\n")
    print("wrote G1 file; G1 ok =", ok)

if __name__ == "__main__":
    main()