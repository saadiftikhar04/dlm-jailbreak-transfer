#!/bin/bash
# Migrate 4 PENDING diffucoder ArrAttack shards from saturated nvidia pool to
# ebrainccs condo (A100-80G, MaxTRES gpu=4). nvidia-xxl QoS is not attachable
# (Invalid qos specification), so ebrainccs is the free-capacity pool.
set -uo pipefail
ROOT=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
cd "$ROOT" || exit 1
for s in 01 02 03 04; do
  name="arr_s${s}_diffucoder"
  # 1. cancel any stale copy of this shard name (PENDING or old)
  scancel --name="$name" 2>/dev/null
  sleep 2
  # 2. build ebrainccs variant: swap qos + partition
  sed -e 's/--qos=nvidia/--qos=ebrainccs/' \
      -e 's/--partition=nvidia/--partition=condo/' \
      "arr_shard_diffucoder_${s}.sbatch" > "arr_ebrain_${s}_diffucoder.sbatch"
  # 3. submit
  jobid=$(sbatch --parsable "arr_ebrain_${s}_diffucoder.sbatch" 2>&1)
  echo "ebrain shard ${s}: $jobid"
done
echo "=== submitted ebrainccs migrates; queue below ==="
squeue -u bc3194 -o '%.10i %.18j %.2t %.12Q' 2>/dev/null | grep -E 'ebrain|diffucoder'