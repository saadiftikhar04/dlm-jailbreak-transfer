# A5/S2 three-attack comparison on the 165 common prompts (J1 D3 judge)

ITT outcomes from the merged J1 labels; the 165 ArrAttack held-out keys were rebuilt
from stored source rows and verified identical across victims before analysis.

## Cell rates

| victim | attack | successes / 165 | ASR | Wilson 95% | duplicate-cluster 95% |
|---|---|---|---|---|---|
| qwen | pif | 35/165 | 0.2121 | [0.1567, 0.2807] | [0.1515, 0.2788] |
| qwen | metacipher | 127/165 | 0.7697 | [0.6998, 0.8274] | [0.7030, 0.8303] |
| qwen | arrattack | 31/165 | 0.1879 | [0.1356, 0.2543] | [0.1333, 0.2485] |
| llama31_8b | pif | 27/165 | 0.1636 | [0.1150, 0.2276] | [0.1091, 0.2183] |
| llama31_8b | metacipher | 33/165 | 0.2000 | [0.1461, 0.2675] | [0.1394, 0.2606] |
| llama31_8b | arrattack | 16/165 | 0.0970 | [0.0606, 0.1517] | [0.0545, 0.1455] |
| llada | pif | 28/165 | 0.1697 | [0.1201, 0.2343] | [0.1152, 0.2303] |
| llada | metacipher | 13/165 | 0.0788 | [0.0466, 0.1301] | [0.0424, 0.1212] |
| llada | arrattack | 50/165 | 0.3030 | [0.2380, 0.3770] | [0.2364, 0.3758] |
| dream | pif | 1/165 | 0.0061 | [0.0011, 0.0335] | [0.0000, 0.0182] |
| dream | metacipher | 3/165 | 0.0182 | [0.0062, 0.0521] | [0.0000, 0.0424] |
| dream | arrattack | 4/165 | 0.0242 | [0.0095, 0.0607] | [0.0061, 0.0485] |
| diffucoder | pif | 3/165 | 0.0182 | [0.0062, 0.0521] | [0.0000, 0.0424] |
| diffucoder | metacipher | 8/165 | 0.0485 | [0.0248, 0.0927] | [0.0182, 0.0848] |
| diffucoder | arrattack | 13/165 | 0.0788 | [0.0466, 0.1301] | [0.0424, 0.1212] |

## Cochran's Q across the three attacks

| victim | Q | df | p |
|---|---|---|---|
| qwen | 132.0597 | 2 | 2.107e-29 |
| llama31_8b | 7.6897 | 2 | 0.02139 |
| llada | 30.1159 | 2 | 2.887e-07 |
| dream | 1.7500 | 2 | 0.4169 |
| diffucoder | 6.5217 | 2 | 0.03836 |

## Pairwise exact McNemar with Holm (attack-versus-attack family)

| victim | pair | first | second | diff (pp) | 95% CI (pp) | exact p | Holm p |
|---|---|---|---|---|---|---|---|
| qwen | pif>metacipher | 35 | 127 | -55.76 | [-64.85, -46.06] | 1.867e-19 | 2.613e-18 |
| qwen | pif>arrattack | 35 | 31 | +2.42 | [-4.85, +9.70] | 0.6358 | 1 |
| qwen | metacipher>arrattack | 127 | 31 | +58.18 | [+49.09, +66.67] | 1.981e-22 | 2.971e-21 |
| llama31_8b | pif>metacipher | 27 | 33 | -3.64 | [-12.12, +4.85] | 0.4709 | 1 |
| llama31_8b | pif>arrattack | 27 | 16 | +6.67 | [+0.61, +12.73] | 0.04329 | 0.3463 |
| llama31_8b | metacipher>arrattack | 33 | 16 | +10.30 | [+2.42, +18.18] | 0.01372 | 0.1423 |
| llada | pif>metacipher | 28 | 13 | +9.09 | [+1.82, +16.36] | 0.02753 | 0.2478 |
| llada | pif>arrattack | 28 | 50 | -13.33 | [-20.61, -6.06] | 0.0009407 | 0.01129 |
| llada | metacipher>arrattack | 13 | 50 | -22.42 | [-30.30, -13.94] | 4.336e-07 | 5.637e-06 |
| dream | pif>metacipher | 1 | 3 | -1.21 | [-3.64, +1.21] | 0.625 | 1 |
| dream | pif>arrattack | 1 | 4 | -1.82 | [-4.85, +0.61] | 0.375 | 1 |
| dream | metacipher>arrattack | 3 | 4 | -0.61 | [-3.64, +2.42] | 1 | 1 |
| diffucoder | pif>metacipher | 3 | 8 | -3.03 | [-7.27, +0.61] | 0.2266 | 1 |
| diffucoder | pif>arrattack | 3 | 13 | -6.06 | [-10.30, -1.82] | 0.01294 | 0.1423 |
| diffucoder | metacipher>arrattack | 8 | 13 | -3.03 | [-8.48, +2.42] | 0.3833 | 1 |
