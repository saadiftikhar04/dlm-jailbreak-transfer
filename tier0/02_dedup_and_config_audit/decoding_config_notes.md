# T02.3 Decoding-config audit (AUTHORITATIVE, from generation code)

Concern C7. Victim decoding was **not** held constant across attacks or across families. Every value is read from the generation code (provenance column in decoding_config_matrix_authoritative.csv). Five inconsistencies matter:

**1. LLaDA denoising differs 4x across attacks (the Appendix A.3 issue).**
- PiF: `steps = gen_length`, `block_length = 128` -> **128 steps per block**.
- ArrAttack & MetaCipher: `steps = 128` fixed, `gen_length = 512`, `block_length = 128` -> **32 steps per block**.
So PiF denoises LLaDA four times more thoroughly per block than the other two attacks. LLaDA is the high-ASR diffusion model that drives the family effect, so its cross-attack numbers are confounded by decoding, not just by attack strength.

**2. Dream uses a different algorithm under MetaCipher.**
- PiF / ArrAttack: `alg=origin`, `temp=0.0`, `steps=gen_length`.
- MetaCipher: `alg=entropy`, `temp=0.2`, `steps=128`.
A different decoding algorithm across attacks makes Dream's cross-attack ASR non-comparable.

**3. DiffuCoder temperature/steps differ across attacks.**
- PiF / ArrAttack: `temp=0.3`, `steps=gen_length`. MetaCipher: `temp=0.4`, `steps=128`.

**4. Causal victims are sampled, diffusion victims are (near-)greedy — within every attack.**
- Causal (qwen/llama/falcon): `do_sample=True`, `top_p=0.9`, `temp=0.7-0.8`.
- LLaDA / Dream: `temp=0` (greedy). DiffuCoder: `temp=0.3-0.4`.
So even inside a single attack, the causal-vs-diffusion comparison mixes a stochastic decoder against a near-deterministic one. Sampling temperature alone can move ASR, so part of the family gap is a decoding artifact.

**5. Causal victim temperature differs across attacks.**
- PiF / ArrAttack: `temp=0.8`. MetaCipher: `temp=0.7`. Minor, but another uncontrolled axis.

## Implication
The paper's cross-attack and cross-family ASR comparisons are confounded by victim decoding. At minimum the paper must (a) report this matrix honestly in the appendix, and (b) either re-run a decoding-matched control (T17) or restrict mechanism claims (MEE/MDG) to cells decoded identically. Note LLaDA's 40.3-pt mechanism-dependence gap spans PiF (128 steps/block) and MetaCipher (32 steps/block), so that gap partly measures a 4x denoising difference.

(Guardrail G4: this audit documents the decoding differences; it does not change any decoding setting to hit a target.)
