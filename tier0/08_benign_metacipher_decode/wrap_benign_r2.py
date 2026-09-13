"""R2 Step 15: wrap the 150-prompt benign set with the REAL MetaCipher wrapper.
Faithfully reuses metacipher_multi.KeywordAgent.find_keywords +
CipherPool.generate_ciphered_prompt (cipher=substitution). Records per prompt
the keyword/mask count. Uses DEEPSEEK_API_KEY from env (CPU, API-bound).

Output: benign_metacipher_wrapped_r2.json (150 rows, indent=4).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
os.environ.setdefault("DEEPSEEK_API_KEY", "")
import metacipher_multi as mm

CIPHER = "substitution"
HERE = os.path.dirname(os.path.abspath(__file__))
TO = os.path.join(HERE, "..", "08_benign_metacipher_decode")
OUT = os.path.join(HERE, "..", "08_benign_metacipher_decode", "benign_metacipher_wrapped_r2.json")

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    sys.exit("Set DEEPSEEK_API_KEY")
ka = mm.KeywordAgent(api_key=api_key)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

with open(os.path.join(TO, "benign_prompts_r2.json")) as f:
    benign = json.load(f)
assert len(benign) == 150

wrapped = []
for b in benign:
    prompt = b["prompt"]
    keywords = [k for k in ka.find_keywords(prompt) if k and k.lower() != "none"]
    adv, _ = mm.CipherPool.generate_ciphered_prompt(CIPHER, keywords, prompt, None)
    nmask = adv.count("[MASK")
    wrapped.append({"id": b["id"], "category": b["category"],
                    "original_prompt": prompt, "wrapped_prompt": adv,
                    "keywords": keywords, "n_masks": nmask,
                    "is_new": b["is_new"]})

with open(OUT, "w") as f:
    json.dump(wrapped, f, indent=4, ensure_ascii=False)

from collections import Counter
masked = Counter(1 if w["n_masks"] > 0 else 0 for w in wrapped)
print(f"wrapped {len(wrapped)} benign prompts (cipher={CIPHER})")
print(f"  with >=1 mask: {masked.get(1,0)}, without: {masked.get(0,0)}")
print(f"wrote {OUT}")