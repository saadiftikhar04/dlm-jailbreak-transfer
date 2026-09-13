#!/bin/bash
# Poll remaining T07 victims (llada dream diffucoder); pull each when done.
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
LOCAL=/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
MODELS=(llada dream diffucoder)
mkdir -p $LOCAL/model_outputs
for round in $(seq 1 150); do
  pulled_any=0
  for m in "${MODELS[@]}"; do
    if [ -s "$LOCAL/model_outputs/${m}_outputs.json" ]; then continue; fi
    out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
      "ls -s $BASE/model_outputs/${m}_outputs.json 2>/dev/null && cat $BASE/model_outputs/${m}_outputs.json 2>/dev/null | wc -c" 2>/dev/null)
    if echo "$out" | grep -q "$m"; then
      scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:$BASE/model_outputs/${m}_outputs.json \
        $LOCAL/model_outputs/ 2>/dev/null && echo "PULLED ${m} at $(date)" && pulled_any=1
    fi
  done
  # all three local?
  missing=0
  for m in "${MODELS[@]}"; do [ -s "$LOCAL/model_outputs/${m}_outputs.json" ] || missing=1; done
  if [ "$missing" == "0" ]; then echo "ALL 3 DONE $(date)"; exit 0; fi
  sleep 60
done
echo "POLL_END"