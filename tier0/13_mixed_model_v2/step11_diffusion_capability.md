# R2 Step 11 — symmetric capability controls (diffusion victims)

Existing-output diagnosis (T07/T08/T09, no new generation). Per model:

| model | benign-PiF answer | plaintext-harmful compliance | benign-decode | reading |
|---|---|---:|---:|---|
| dream | 85.0% (51/60) | 3.0% | 36.67% | responds to benign AND can produce harmful on plaintext -> low attacked ASR is attack/judge-specific |
| diffucoder | 96.7% (58/60) | 16.0% | 33.33% | responds to benign AND can produce harmful on plaintext -> low attacked ASR is attack/judge-specific |
| llada | 100.0% (60/60) | 30.0% | 41.67% | responds to benign AND can produce harmful on plaintext -> low attacked ASR is attack/judge-specific |
| falcon | 100.0% (60/60) | 0.0% | 18.33% | mixed/ambiguous |
| llama | 100.0% (60/60) | 20.0% | 16.67% | responds to benign AND can produce harmful on plaintext -> low attacked ASR is attack/judge-specific |
| qwen | 100.0% (60/60) | 24.0% | 60.0% | responds to benign AND can produce harmful on plaintext -> low attacked ASR is attack/judge-specific |

## Honest interpretation
The diffusion victims DO produce non-empty responses to benign prompts (T09: dream 51/60, llada 60/60, diffucoder 58/60) and DO comply with some plaintext harmful prompts (T07: dream 3%, llada 30%, diffucoder
 16%). Their low MetaCipher/ArrAttack ASR is therefore not because they
 emit nothing: it is attack- or judge-specific. This differs from Falcon,
 which emits no final answer on ~71-83% of open-ended prompts; the
 diffusion victims do not have that answer-ceiling confound.

## n~200 caveat
The reviewer wants n~200 for +/-5pp. Current benign sets are n=60 and
plaintext n=100. A full symmetric n~200 rerun for the 3 diffusion victims
across benign + safety-adjacent + open-ended = ~3 x 3 x 200 = ~1,800
diffusion generations (128-step each). That is the explicit GPU budget
item for a compliant symmetric control; the evidence above is the
current honest state.
