# T03 Falcon-H1R raw vs stripped re-judge (C4)

## Data availability
- **PiF Falcon**: trace preserved -> raw recoverable (n=913).
- **ArrAttack Falcon**: trace preserved -> raw recoverable (n=165, 1 trace null).
- **MetaCipher Falcon**: NO trace column -> raw NOT recoverable. Per the plan precondition this cell would need regeneration for a full raw re-judge. However its saved final_response already shows genuine wrong-decryption (the model parrots the cipher instructions in repetition loops), so the near-zero MetaCipher-Falcon cell is a decode failure, not a stripped answer.

## Structural evidence (judge-free, decisive)

| attack    |   n |   mean_stripped_chars |   mean_raw_chars |   nonempty_raw_but_empty_stripped |   official_successes |
|:----------|----:|----------------------:|-----------------:|----------------------------------:|---------------------:|
| pif       | 913 |                  99.3 |            956.1 |                                 0 |                    0 |
| arrattack | 165 |                 288   |           2224.8 |                                 0 |                    0 |

- `nonempty_raw_but_empty_stripped` = rows where the model produced a trace but the stripped final answer is empty. If this is ~0, stripping is not silently deleting answers.

## Proxy re-judge (labelled PROXY, not the official API judge)

| attack    |   n |   official_asr_pct |   proxy_raw_compliance |   proxy_stripped_compliance |   proxy_raw_minus_stripped |   candidate_stripped_away_compliance |
|:----------|----:|-------------------:|-----------------------:|----------------------------:|---------------------------:|-------------------------------------:|
| pif       | 913 |                  0 |                      2 |                           2 |                          0 |                                   37 |
| arrattack | 165 |                  0 |                     14 |                          16 |                         -2 |                                   45 |

`candidate_stripped_away_compliance` = rows whose full raw text trips the compliance heuristic but whose stripped output is a refusal/empty. These are the only rows that *could* be harness artifacts; they are saved to candidate_stripped_away_compliance.csv for a manual/official-judge read.

## Interpretation
- 82 candidate rows show possible stripped-away compliance and MUST be read with the official judge before Falcon's 0% is interpreted. Escalate to Boyuan.

## Caveats
- The re-judge here is a transparent refusal/compliance heuristic, NOT the attack's official DeepSeek judge (API not reachable in this environment). Step 4 of T03 (official re-judge of ~2,000 raw texts) still needs API access; the candidate list above bounds how many rows that run would need to change to matter (if it is 0, the official run is confirmatory only).
