# T06 Wilson CI notes (C3)

Pairwise cells whose 95% Wilson intervals OVERLAP must not be reported as a ranking. Within-attack model pairs checked below.

- Overlapping within-attack model pairs: **17**

- **ArrAttack Dream vs DiffuCoder (plan's main example):** dream 6.06% [3.32,10.8] vs diffucoder 10.3% [6.53,15.88] -> overlap, not rankable.

Full overlap list:

| attack | model A | model B | A | B |
|---|---|---|---|---|
| pif | qwen | llama | 11.61% [9.69,13.85] | 13.69% [11.61,16.07] |
| pif | falcon | dream | 0.0% [0.0,0.42] | 0.0% [0.0,0.42] |
| pif | llada | diffucoder | 7.01% [5.53,8.85] | 5.37% [4.08,7.02] |
| metacipher | qwen | llama | 71.52% [68.51,74.36] | 71.08% [68.06,73.93] |
| metacipher | falcon | dream | 0.22% [0.06,0.8] | 0.11% [0.02,0.62] |
| metacipher | falcon | diffucoder | 0.22% [0.06,0.8] | 0.88% [0.44,1.72] |
| metacipher | dream | diffucoder | 0.11% [0.02,0.62] | 0.88% [0.44,1.72] |
| arrattack | qwen | llama | 9.09% [5.59,14.46] | 8.48% [5.12,13.74] |
| arrattack | qwen | llada | 9.09% [5.59,14.46] | 8.48% [5.12,13.74] |
| arrattack | qwen | dream | 9.09% [5.59,14.46] | 6.06% [3.32,10.8] |
| arrattack | qwen | diffucoder | 9.09% [5.59,14.46] | 10.3% [6.53,15.88] |
| arrattack | llama | llada | 8.48% [5.12,13.74] | 8.48% [5.12,13.74] |
| arrattack | llama | dream | 8.48% [5.12,13.74] | 6.06% [3.32,10.8] |
| arrattack | llama | diffucoder | 8.48% [5.12,13.74] | 10.3% [6.53,15.88] |
| arrattack | llada | dream | 8.48% [5.12,13.74] | 6.06% [3.32,10.8] |
| arrattack | llada | diffucoder | 8.48% [5.12,13.74] | 10.3% [6.53,15.88] |
| arrattack | dream | diffucoder | 6.06% [3.32,10.8] | 10.3% [6.53,15.88] |
