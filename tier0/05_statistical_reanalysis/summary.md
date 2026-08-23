# T05 Statistical re-analysis (C6)

## 1. 18-cell ASR matrix

| model      |   arrattack |   metacipher |   pif |
|:-----------|------------:|-------------:|------:|
| diffucoder |       10.3  |         0.88 |  5.37 |
| dream      |        6.06 |         0.11 |  0    |
| falcon     |        0    |         0.22 |  0    |
| llada      |        8.48 |        47.32 |  7.01 |
| llama      |        8.48 |        71.08 | 13.69 |
| qwen       |        9.09 |        71.52 | 11.61 |

## 2. Full ANOVA — the missing variance is now named

The current paper reports attack 27.9%, family 6.1%, model-within-family 55.3% (sum 89.3%) and never names the remaining 10.7%. The full saturated decomposition attributes every point:

| term                                            |       SS |   df |   pct_of_total |
|:------------------------------------------------|---------:|-----:|---------------:|
| attack                                          | 2537.91  |    2 |          27.92 |
| family                                          |  557.504 |    1 |           6.13 |
| model_within_family                             | 2508.77  |    4 |          27.6  |
| attack_x_family                                 |  968.268 |    2 |          10.65 |
| attack_x_model_within_family (THE MISSING TERM) | 2518.38  |    8 |          27.7  |
| TOTAL                                           | 9090.83  |   17 |         100    |

- **attack x model-within-family = 27.7% of total variance** is the term the current decomposition drops. This interaction is exactly the paper's qualitative thesis (attacks behave differently across models), so it must be named, not hidden in a residual.
- attack x family interaction = 10.65%.

## 3. Case-level logistic mixed model

Fit on all 11,946 binary outcomes: `success ~ attack*family + (1|model) + (1|prompt)`. This handles the zero boundary, unequal denominators (913 vs 165), and the repeated use of each prompt across models. See mixed_model_summary.txt. ANOVA on raw percentages near the zero boundary (the current approach) is the wrong model; the mixed model is the defensible one.

## 4/5. Leave-one-model-out + FSR

| dropped    |   causal |   diffusion |   gap_causal_minus_diffusion |   FSR_diffusion_over_causal |
|:-----------|---------:|------------:|-----------------------------:|----------------------------:|
| NONE       |   0.2618 |      0.0996 |                       0.1622 |                      0.3804 |
| qwen       |   0.1984 |      0.0996 |                       0.0988 |                      0.5021 |
| llama      |   0.1949 |      0.0996 |                       0.0953 |                      0.5112 |
| falcon     |   0.3923 |      0.0996 |                       0.2927 |                      0.2539 |
| llada      |   0.2618 |      0.0213 |                       0.2405 |                      0.0815 |
| dream      |   0.2618 |      0.1467 |                       0.1152 |                      0.5601 |
| diffucoder |   0.2618 |      0.1308 |                       0.131  |                      0.4997 |

- **FSR sensitivity:** dropping a single victim swings the family gap and FSR materially (see the `dropped` rows). With one near-zero model in each family (Falcon causal; Dream diffusion), **FSR<1 is close to forced by model selection**, not discovered. RQ2 must be restated as conditional on these six models.

FSR by attack:

| attack     |   causal_asr |   diffusion_asr |    FSR |
|:-----------|-------------:|----------------:|-------:|
| pif        |       0.0843 |          0.0413 | 0.4892 |
| metacipher |       0.4761 |          0.161  | 0.3382 |
| arrattack  |       0.0586 |          0.0828 | 1.4138 |
| ALL        |       0.2618 |          0.0996 | 0.3804 |

## 6. Per-model 3-vector cosine similarity

| model      |   diffucoder |   dream |   falcon |   llada |   llama |   qwen |
|:-----------|-------------:|--------:|---------:|--------:|--------:|-------:|
| diffucoder |       1      |  0.8854 |   0.0755 |  0.2944 |  0.2631 | 0.2573 |
| dream      |       0.8854 |  1      |   0.0181 |  0.1922 |  0.134  | 0.1422 |
| falcon     |       0.0755 |  0.0181 |   1      |  0.974  |  0.9753 | 0.9794 |
| llada      |       0.2944 |  0.1922 |   0.974  |  1      |  0.9974 | 0.9986 |
| llama      |       0.2631 |  0.134  |   0.9753 |  0.9974 |  1      | 0.9995 |
| qwen       |       0.2573 |  0.1422 |   0.9794 |  0.9986 |  0.9995 | 1      |

MetaCipher-dominated vectors sit in the positive orthant and are near-collinear by construction, so high cosine similarity between two such models is not independent evidence of a shared 'mechanism'.

## Interpretation rule (from the plan)
- If the family effect flips sign or loses significance when Falcon-H1R is dropped, the paper must say so and RQ2 becomes conditional on the specific six models. The LOMO table above is the direct check.
