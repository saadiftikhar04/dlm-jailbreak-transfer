# T02.2 Response-length statistics (FINAL, true tokens)

Token lengths computed with each victim's REAL tokenizer on the HPC (all-mpnet-style counts were a space-split proxy that under-counts by 1.2-3.9x and hid the signal). Generation cap in code = 512 max_new_tokens for all attacks.

## Core finding (contradicts a naive length-artifact story)
- Metacipher causal victims produce LONG responses: qwen mean=539 tok, llama=1010, falcon=974, i.e. the 512 cap does NOT hold them down.
- Metacipher diffusion victims are SHORT: dream mean=43 tok, diffucoder=63, llada=495. They are not 'capped shorter'; they mostly refuse or answer briefly.
- Even single-attempt (total_attempts=1) qwen responses reach 5494 chars (~1300 tokens), so these are complete compliant answers, not truncated scraps.
- Conclusion: the low diffusion ASR is NOT a length artifact. Diffusion responds less because it decodes/refuses poorly (matches T01: dream dominated by wrong_decryption, diffucoder by too_general).

## Which cells sit at/near the 512 cap (median >= 512)
cap_flag_strict
False    15
True      3

Cells where median token length reaches/exceeds 512 (likely truncated or full-capacity answers):
    attack  model  mean_tok  median_tok  truncation_rate_pct
metacipher   qwen     538.5       527.0                56.96
metacipher  llama    1010.1      1024.0                99.45
metacipher falcon     973.5      1023.0                96.28

## Caveats (G3 / honesty)
- 'rows_at_cap_512' counts responses whose true-token length >= 512. For metacipher causal this includes complete multi-step answers, so it is NOT a truncation count by itself; use cap_flag_strict (median >= 512) for the conservative 'pinned at cap' interpretation.
- empty_rate=0 for every cell: there are no blank final responses saved (empty/malformed are either absent or were filtered before the judged CSV). Verified directly from the judged files rather than assumed.
- The LLaDA cross-attack gap (T02.3) is a real decoding difference (128 vs 32 steps/block), not a token-length artifact.
