# A3 outcome mechanism audit, C0 result

Generated from existing files only. No generation, API call, or new judging was run. The source hashes and reproducible builder are adjacent to this note. No prompt or response text is emitted by the artifacts.

## Existing four-way MetaCipher outcomes

`metacipher_outcomes_by_model_benchmark.csv` rebuilds the official four-way counts from all 24 model-by-benchmark slices. These are the existing attack-specific categories, not independent semantic labels or ground truth. PiF has only binary labels, and ArrAttack has its separate three-way categorizer, so those cells are not relabeled as four-way outcomes.

## Fixed placeholder detector

The lexical detector found 3209 of 11946 responses with at least one fixed placeholder marker. Pattern-level matches (overlap allowed): `{"explicit_angle_placeholder": 194, "explicit_bracket_placeholder": 3097, "explicit_curly_placeholder": 0, "placeholder_instruction_phrase": 406}`. Per-cell counts are in `placeholder_hits_by_cell.csv`. This is a screening count only; precision/recall and false-positive rates are unknown until a blinded validation sample is scored.

The four frozen patterns detect explicit angle-, square-bracket-, and curly-brace placeholders plus a short list of explicit template phrases. No pattern was tuned after reviewing cell rates. A hit is not automatically evidence that the entire response is unusable.
The square-bracket rule deliberately flags literal `[MASKn]` markers; the 3,097 bracket-pattern matches may therefore include cipher-token mentions rather than unfinished answer templates. Treat the 3,209 total as candidate markers, not a confirmed placeholder prevalence estimate, until blinded validation distinguishes these cases.

## Existing benign MetaCipher control

The saved control has 150 prompts, 900 judgments, 2 prompts at least 126 characters long, and 50 prompts ending in a common function word. Existing labels and deterministic prompt-prefix echo counts are split by victim and masked status in `benign_control_existing_labels.csv`.

Among the existing `632` `decode_and_comply` labels, 117 responses are exact normalized prompt prefixes of at least 40 characters. This demonstrates that the historical label can include substantial prompt echo rather than a completed answer. The check is lexical only and does not assess semantic correctness.

The historic four-way control labels do not independently score semantic recovery, completion of the benign task, and answer quality. In particular, the old `decode_and_comply` field combines decoding and compliance, while `malformed_or_empty` combines malformed output, empty output, and a pure template response. Therefore the apparent masked-versus-unmasked compliance difference cannot by itself identify either successful semantic recovery or successful fulfillment of the benign request. A valid direct semantic-recovery estimate requires an approved row-level reassessment against the original unwrapped request, with separate decode and answer-completion fields.

## Conclusion and remaining gate

The four-way MetaCipher distribution describes observed judge outcomes. For Dream and DiffuCoder, wrong-decryption and too-general counts are evidence of response-level failure modes, but they do not establish the latent reason for a specific response. LLaDA remains a contrast cell. The benign control supports a decoding-capability question only to the extent its existing labels are trustworthy; prompt-echo checks show that `decode_and_comply` is not a clean task-completion measure. Do not claim that the benign control proves refusal after decoding or resolves the exact harmful-outcome mechanism.

C0 is complete. The plan's direct semantic-recovery check is unresolved and needs explicit approval for a frozen, prompt-level reassessment before any API judge batch is run. D5 human calibration remains separately governed by consent, institutional review, and the approval gates; no human labeling was performed here.
