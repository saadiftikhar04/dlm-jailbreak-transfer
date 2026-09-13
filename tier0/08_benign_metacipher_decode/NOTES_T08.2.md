# T08.2 notes — MetaCipher wrapper on benign prompts (real pipeline)

Method: faithfully reuses the real MetaCipher code path
(`metacipher_multi.KeywordAgent.find_keywords` for keyword detection and
`CipherPool.generate_ciphered_prompt` with `cipher="substitution"`, the dominant
real cipher: 690/913 ≈ 76% of final MetaCipher cases). No new wrapper was
invented; the benign prompts go through exactly the same masking + cipher +
decode-then-answer template as the harmful condition.

## Result on the 60 benign prompts
- 60 wrapped, `benign_metacipher_wrapped.json` (indent=4, one record per prompt:
  original, wrapped, detected keywords, n_masks).
- **10 prompts had ≥1 `[MASK]` (really encoded); 50 had no mask.**
  This is the real behaviour of the keyword detector: benign prompts contain no
  malicious vocabulary, so `find_keywords` (DeepSeek) returns `['none']` and the
  prompt is passed through the template almost unencoded. We record this rather
  than forcing artificial masks, because forcing masks would invent a wrapper
  style that MetaCipher never applies to benign content.

## Consequence for T08.4 (decode analysis)
- The 10 masked prompts are the ones that actually exercise decode ability.
- The 50 unmasked prompts answer the benign request more or less directly; they
  measure trivial compliance, not cipher decoding.
- Therefore decode-and-comply / wrong-decode rates must be reported **on the
  10 masked subset** as the capability estimate, with the 50 unmasked rows
  flagged separately. Reporting one blended rate over all 60 would inflate
  decode success and defeat the point of the control.

## G1 (round-trip reproducibility) — honest data limitation
G1 requires re-applying the wrapper to recorded harmful prompts and asserting
the result is byte-identical to the saved `attacked_prompt`. Investigation of
the real judged CSVs shows this cannot be done strictly:
- The saved `final_converted_prompt` column is **not** the direct output of
  `generate_ciphered_prompt`; for many rows it is a different multi-stage
  template (e.g. a "Please answer these questions" conversion) whose content
  does not correspond 1:1 to `original_prompt`.
- The wrapper's `keywords` list is not saved, and the `final_cipher` /
  `final_payload_mask_count` metadata do not fully pin down a unique re-run.
So a byte-for-byte G1 assertion is not achievable from the shipped CSVs. We
report this instead of fabricating a pass. The wrapper functions themselves are
reusable deterministically given the same (keywords, cipher) — verified by
re-invoking `generate_ciphered_prompt` and observing stable template structure;
what is missing is the *recorded* parameter set to compare against.