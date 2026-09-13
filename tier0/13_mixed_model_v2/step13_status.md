# R2 Step 13 — full-pool ArrAttack: RUNNING status (2026-09-05 ~23:30 GST)

## Deployed and running
Self-contained HPC env: /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/
(created because student /scratch/si2356 is unreadable; built our own utils,
913-prompt dataset from pool_913.csv, stage5_fullpool.py, 6 per-victim sbatch).

5/6 victims RUNNING on HPC (1 GPU each, nvidia A100, MAX_ATTEMPTS=50,
stage5-native resume-safe):
  dream, qwen2.5, llama, llada, diffucoder   (qwen/llama causal fast; 3 diffusion slow)
falcon: blocked on HPC (raven env mamba_ssm ABI-broken under torch 2.13+cu130,
transformers.FalconH1 class import crashes). Must run locally. Local GPU currently
full (user's other jobs), so a falcon_gpu_waiter.sh polls for >=18GB free and
auto-launches falcon fullpool from a prebuilt falcon_env/.

## Issues fixed during rollout
1. si2356 paths -> bc3194 self-contained env (qwen_utils, model_utils, stage5).
2. Missing deps downloaded: humarin/T5 (5.5GB), hubert233/GPTFuzz (~0.4GB),
   sentence-transformers/all-mpnet (~0.4GB), meta-llama/Llama-3.1-8B (16GB).
3. Llama gated-repo offline: transformers tried online token revalidation even
   with local_files_only -> failed. FIX: point model_utils REGISTRY["llama"] at
   the absolute snapshot dir. Reloaded fine. (Verified: llama now RUNNING.)
4. Falcon mamba ABI: empty mamba_ssm stub did NOT work (FalconH1ForCausalLM class
   not exported when mamba import path changes) -> falcon is local-only here.

## Realistic timing (measured, must be surfaced)
From llada progress CSV: prompts cost 100-500s each (15/11/32/50 attempts).
diffucoder first prompt ~28 min (41 attempts, still going). Estimates:
  - causal (qwen, llama): ~1-2s/target x up to 50 attempts -> hours per victim
  - llada: ~3-5 min/prompt avg -> 913 x ~4 min ≈ 60-75 h (~3 days)
  - dream: ~5-13 attempts avg -> 70-110 h
  - diffucoder: ~20-50 attempts avg -> 120-240 h (5-10 days)
This matches the plan's "largest single compute item". All 5 are progressing in
parallel; completion is days, not hours, for the diffusion victims.

## Decision log (user confirmed, 2026-09-05)
1. All 6 victims (not just diffusion).  2. Accept budget, shards to save time.
3. Do NOT reduce MAX_ATTEMPTS (keep faithful 50).

## After all victims finish
step13_aggregate.py reads each results/<key>/arrattack_progress.csv -> per-victim
+ per-dataset ASR with Wilson CIs; write into Section 5 (ArrAttack now full-pool
denominator 913 matching PiF/MetaCipher; keep 165 as secondary row).
Set ARR_FULLPOOL_RESULTS=/scratch/.../results before running it.

## Files
HPC: arrattack_fullpool/{stage5_fullpool.py, utils/, step13_dataset/,
arr_full_*.sbatch, results/<key>/, hpc_progress.sh, mamba_stub/, falcon_env/}
Local: tier0/13_mixed_model_v2/{step13_*, arr_full_*.sbatch, falcon_env/,
falcon_gpu_waiter.sh, step13_aggregate.py}