#!/bin/bash
# R2 Step 13 — first-run verification: launch ONE victim (dream) to verify the
# full pipeline (rewrite -> victim gen -> GPTFuzz/DeepSeek judge) end-to-end
# before batch-submitting the remaining 5 victims. stage5 is resume-safe via its
# PROGRESS_CSV, so any per-prompt error is skipped on re-run rather than fatal.
# Usage: sbatch arr_full_dream.sbatch   (after deps downloaded)
echo "Run arr_full_dream.sbatch first; check its .out/.err for clean prompt
processing (no import errors, GPTFuzz/T5 loaded, victims transferred). Once
clean, submit the other 5 arr_full_*.sbatch."