#!/bin/bash
# Poll T12 generation jobs (dream/diffucoder/llada). Pulls a file ONLY when the
# HPC-side JSONL line count reaches the target for that model (avoids pulling a
# half-written file, which a naive scp-on-nonempty would grab and then skip).
# Dream has fewer sample rows (B arm limited), hence lower target line count.
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/12_low_asr_multiseed/model_outputs
LOCAL=/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/12_low_asr_multiseed/model_outputs
# model -> target JSONL line count (from multiseed_sample.json)
declare -A TARGET=( [dream]=26 [diffucoder]=30 [llada]=30 )
MODELS=(dream diffucoder llada)
mkdir -p $LOCAL
for round in $(seq 1 300); do
  pulled=0
  for m in "${MODELS[@]}"; do
    tgt=${TARGET[$m]}
    for seed in 1 2 3; do
      f="${m}_seed${seed}_outputs.json"
      # skip if local already complete
      if [ -s "$LOCAL/$f" ] && [ "$(wc -l < "$LOCAL/$f")" -ge "$tgt" ]; then continue; fi
      n=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
        "wc -l < $BASE/$f 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
      if [ -n "$n" ] && [ "$n" -ge "$tgt" ] 2>/dev/null; then
        scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:$BASE/$f $LOCAL/ 2>/dev/null && \
          { echo "PULLED $f ($n/$tgt) at $(date)"; pulled=1; }
      fi
    done
  done
  missing=0
  for m in "${MODELS[@]}"; do
    tgt=${TARGET[$m]}
    for seed in 1 2 3; do
      [ -s "$LOCAL/${m}_seed${seed}_outputs.json" ] && [ "$(wc -l < "$LOCAL/${m}_seed${seed}_outputs.json")" -ge "$tgt" ] || missing=1
    done
  done
  if [ "$missing" == "0" ]; then echo "ALL DONE $(date)"; exit 0; fi
  sleep 120
done
echo "POLL_END_TIMEOUT $(date)"