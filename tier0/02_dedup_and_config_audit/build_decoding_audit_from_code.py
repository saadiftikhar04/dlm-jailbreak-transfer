"""
T02.3 (COMPLETED with generation code): authoritative victim-decoding matrix.
Concern C7 (non-comparability). Every value below is read from the generation
code in scripts.zip; provenance (file:line) is recorded per row. No data-derived
guessing. Guardrail G4: we report that decoding was NOT held constant; we do not
change it.

Sources:
  PiF        : scripts/pif_target_models.py
  ArrAttack  : scripts/stage5_attack.py + utils/model_utils.py
  MetaCipher : scripts/metacipher_multi.py
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "02_dedup_and_config_audit")

# code-confirmed victim decoding settings (see provenance column)
rows = [
    # PiF
    ("pif", "qwen", "causal", "sampling", "temp=0.8, top_p=0.9, do_sample=True",
     "steps=n/a", "pif_target_models.py:208-211"),
    ("pif", "llama", "causal", "sampling", "temp=0.8, top_p=0.9, do_sample=True",
     "steps=n/a", "pif_target_models.py:208-211"),
    ("pif", "falcon", "causal", "sampling", "temp=0.8, top_p=0.9, do_sample=True",
     "steps=n/a", "pif_target_models.py:208-211"),
    ("pif", "llada", "diffusion", "greedy-diffusion",
     "temp=0, remasking=low_confidence", "steps=gen_length, block=128 => 128 steps/block",
     "pif_target_models.py:169-170"),
    ("pif", "dream", "diffusion", "diffusion-origin", "temp=0.0, top_p=None, alg=origin",
     "steps=gen_length", "pif_target_models.py:126-128"),
    ("pif", "diffucoder", "diffusion", "diffusion-entropy", "temp=0.3, top_p=0.95, alg=entropy",
     "steps=gen_length", "pif_target_models.py:145-147"),
    # ArrAttack
    ("arrattack", "qwen", "causal", "sampling", "temp=0.8, top_p=0.9, do_sample=True",
     "steps=n/a", "utils/model_utils.py:157-160"),
    ("arrattack", "llama", "causal", "sampling", "temp=0.8, top_p=0.9, do_sample=True",
     "steps=n/a", "utils/model_utils.py:157-160"),
    ("arrattack", "falcon", "causal", "sampling", "temp=0.8, top_p=0.9, do_sample=True",
     "steps=n/a", "utils/model_utils.py:157-160"),
    ("arrattack", "llada", "diffusion", "greedy-diffusion",
     "temp=0, remasking=low_confidence", "steps=128 fixed, gen_length=512, block=128 => 32 steps/block",
     "stage5_attack.py:276,377-382"),
    ("arrattack", "dream", "diffusion", "diffusion-origin", "temp=0.0, top_p=None, alg=origin",
     "steps=gen_length", "stage5_attack.py:399-415"),
    ("arrattack", "diffucoder", "diffusion", "diffusion-entropy", "temp=0.3, top_p=0.95, alg=entropy",
     "steps=gen_length", "stage5_attack.py:416-434"),
    # MetaCipher
    ("metacipher", "qwen", "causal", "sampling", "temp=0.7, top_p=0.9, do_sample=True",
     "steps=n/a", "metacipher_multi.py:1088-1091"),
    ("metacipher", "llama", "causal", "sampling", "temp=0.7, top_p=0.9, do_sample=True",
     "steps=n/a", "metacipher_multi.py:1088-1091"),
    ("metacipher", "falcon", "causal", "sampling", "temp=0.7, top_p=0.9, do_sample=True",
     "steps=n/a", "metacipher_multi.py:1088-1091"),
    ("metacipher", "llada", "diffusion", "greedy-diffusion",
     "temp=0, remasking=low_confidence", "steps=128 fixed, gen_length=512, block=128 => 32 steps/block",
     "metacipher_multi.py:129,1137"),
    ("metacipher", "dream", "diffusion", "diffusion-entropy", "temp=0.2, top_p=0.95, alg=entropy",
     "steps=128 fixed", "metacipher_multi.py:1162-1164"),
    ("metacipher", "diffucoder", "diffusion", "diffusion-entropy", "temp=0.4, top_p=0.95, alg=entropy",
     "steps=128 fixed", "metacipher_multi.py:1189-1191"),
]
cfg = pd.DataFrame(rows, columns=["attack", "model", "family", "mode",
                                  "sampling_params", "diffusion_schedule", "provenance"])
cfg.to_csv(os.path.join(OUT, "decoding_config_matrix_authoritative.csv"), index=False)

notes = [
    "# T02.3 Decoding-config audit (AUTHORITATIVE, from generation code)\n",
    "Concern C7. Victim decoding was **not** held constant across attacks or across "
    "families. Every value is read from the generation code (provenance column in "
    "decoding_config_matrix_authoritative.csv). Five inconsistencies matter:\n",
    "**1. LLaDA denoising differs 4x across attacks (the Appendix A.3 issue).**",
    "- PiF: `steps = gen_length`, `block_length = 128` -> **128 steps per block**.",
    "- ArrAttack & MetaCipher: `steps = 128` fixed, `gen_length = 512`, "
    "`block_length = 128` -> **32 steps per block**.",
    "So PiF denoises LLaDA four times more thoroughly per block than the other two "
    "attacks. LLaDA is the high-ASR diffusion model that drives the family effect, so "
    "its cross-attack numbers are confounded by decoding, not just by attack strength.\n",
    "**2. Dream uses a different algorithm under MetaCipher.**",
    "- PiF / ArrAttack: `alg=origin`, `temp=0.0`, `steps=gen_length`.",
    "- MetaCipher: `alg=entropy`, `temp=0.2`, `steps=128`.",
    "A different decoding algorithm across attacks makes Dream's cross-attack ASR "
    "non-comparable.\n",
    "**3. DiffuCoder temperature/steps differ across attacks.**",
    "- PiF / ArrAttack: `temp=0.3`, `steps=gen_length`. MetaCipher: `temp=0.4`, "
    "`steps=128`.\n",
    "**4. Causal victims are sampled, diffusion victims are (near-)greedy — within "
    "every attack.**",
    "- Causal (qwen/llama/falcon): `do_sample=True`, `top_p=0.9`, `temp=0.7-0.8`.",
    "- LLaDA / Dream: `temp=0` (greedy). DiffuCoder: `temp=0.3-0.4`.",
    "So even inside a single attack, the causal-vs-diffusion comparison mixes a "
    "stochastic decoder against a near-deterministic one. Sampling temperature alone "
    "can move ASR, so part of the family gap is a decoding artifact.\n",
    "**5. Causal victim temperature differs across attacks.**",
    "- PiF / ArrAttack: `temp=0.8`. MetaCipher: `temp=0.7`. Minor, but another "
    "uncontrolled axis.\n",
    "## Implication",
    "The paper's cross-attack and cross-family ASR comparisons are confounded by "
    "victim decoding. At minimum the paper must (a) report this matrix honestly in "
    "the appendix, and (b) either re-run a decoding-matched control (T17) or restrict "
    "mechanism claims (MEE/MDG) to cells decoded identically. Note LLaDA's 40.3-pt "
    "mechanism-dependence gap spans PiF (128 steps/block) and MetaCipher (32 "
    "steps/block), so that gap partly measures a 4x denoising difference.",
    "\n(Guardrail G4: this audit documents the decoding differences; it does not "
    "change any decoding setting to hit a target.)",
]
with open(os.path.join(OUT, "decoding_config_notes.md"), "w") as f:
    f.write("\n".join(notes) + "\n")

print("=== T02.3 authoritative decoding matrix ===")
print(cfg.to_string(index=False))
print("\nWrote decoding_config_matrix_authoritative.csv + decoding_config_notes.md")
