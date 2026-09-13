# R2 Step 2 - numerical inconsistency reconciliation

Every before/after below traces to a script and, where possible, a
raw artifact. The ASR source of record for the paper is the official
judge on the FULL pool (Table 1).

## (a) Table 1 (full-pool official) vs Table 6 (common-judge) `Official` column

Root cause: Table 6's `Official` column came from the reweighted
2,982-row subsample (per-cell sampling weight \~5.50 on PiF/MetaCipher),
NOT the full 913/165-pool official ASR that Table 1 reports. Two
different best-effort population estimates for the same concept.

| model | attack | Table1 official | Table6 official | delta pp |
|---|---|---:|---:|---:|
| qwen | pif | 11.61 | 10.24 | +1.37 |
| qwen | metacipher | 71.52 | 72.29 | -0.77 |
| qwen | arrattack | 9.09 | 9.09 | +0.0 |
| llama | pif | 13.69 | 13.86 | -0.17 |
| llama | metacipher | 71.08 | 66.87 | +4.21 |
| llama | arrattack | 8.48 | 8.48 | +0.0 |
| falcon | pif | 0.0 | 0.0 | +0.0 |
| falcon | metacipher | 0.22 | 0.0 | +0.22 |
| falcon | arrattack | 0.0 | 0.0 | +0.0 |
| llada | pif | 7.01 | 5.42 | +1.59 |
| llada | metacipher | 47.32 | 51.81 | -4.49 |
| llada | arrattack | 8.48 | 8.48 | +0.0 |
| dream | pif | 0.0 | 0.0 | +0.0 |
| dream | metacipher | 0.11 | 0.0 | +0.11 |
| dream | arrattack | 6.06 | 6.06 | +0.0 |
| diffucoder | pif | 5.37 | 3.61 | +1.76 |
| diffucoder | metacipher | 0.88 | 0.6 | +0.28 |
| diffucoder | arrattack | 10.3 | 10.3 | +0.0 |

Max absolute discrepancy: 4.49 pp.

**Fix applied:** the common-judge table's `Official` column is now the
FULL-POOL official value (identical to Table 1); the `Common-A`/`Common-B`
columns remain reweighted-subsample estimates and are captioned as such
with the sampling scheme (2,982-row stratified fixed-seed subsample;
PiF/MetaCipher ~166/cell weight ~5.50, ArrAttack all 165/cell weight 1.00).
One ASR source (official, full pool), one secondary judge-boundary view.

## (b) Discussion quotes Dream/DiffuCoder official MetaCipher ASR

| location | Dream MC | DiffuCoder MC |
|---|---|---|
| Section 6.6 (old) | 0.0% | 0.6% |
| Table 1 (full pool) | 0.1% | 0.9% |

**Fix applied:** use the full-pool Table 1 values everywhere
(0.1% / 0.9%).


## (c) Reproduction-audit agreement

Recompute from raw repro JSONs (51 unique rows):
- recorded success, reproduced harmful: 14
- recorded success, reproduced not harmful: 7
- recorded failure, held as failure: 14
- recorded failure, escaped to harmful: 16

**Agreement = 28/51 = 54.9%** (recorded success that
reproduced harmful + recorded failure that held as failure), defined
explicitly as the fraction of sampled rows whose reproduced verdict
matches their recorded verdict under the audit's unified binary judge.
This replaces the stale '21/51 (41%)' figure. The 16 failure->harmful
escapes are, by construction, NOT official successes (0/16); 7 are
audit-judge boundary noise (recorded response also judged harmful on
replay) and 9 are regeneration variance.

