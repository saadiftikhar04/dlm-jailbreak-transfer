#!/bin/bash
cd /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool || exit 1
echo "=== full-pool victims (results/) ==="
for v in qwen2.5 llama llada; do
  f="results/$v/arrattack_progress.csv"
  n=0
  [ -f "$f" ] && n=$(($(wc -l < "$f") - 1))
  echo "$v: $n/913"
done
echo "=== shards (results_shard/) ==="
for v in dream diffucoder; do
  tot=0
  for d in results_shard/${v}_*; do
    f="$d/arrattack_progress.csv"
    [ -f "$f" ] && tot=$((tot + $(wc -l < "$f") - 1))
  done
  echo "$v (shards): $tot/913"
done
echo "=== queue ==="
squeue -u bc3194 -h -o '%j %T' 2>/dev/null | sort | uniq -c