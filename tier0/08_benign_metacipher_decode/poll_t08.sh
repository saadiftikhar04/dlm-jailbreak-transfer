#!/bin/bash
# Poll T08.3 benign MetaCipher generation jobs; pull each completed output.
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/08_benign_metacipher_decode
LOCAL=/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/08_benign_metacipher_decode
MODELS=(qwen llama falcon llada dream diffucoder)
mkdir -p $LOCAL/model_outputs
for round in $(seq 1 200); do
  pulled=0
  for m in "${MODELS[@]}"; do
    if [ -s "$LOCAL/model_outputs/${m}_outputs.json" ]; then continue; fi
    out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
      "ls -s $BASE/model_outputs/${m}_outputs.json 2>/dev/null && cat $BASE/model_outputs/${m}_outputs.json 2>/dev/null | wc -c" 2>/dev/null)
    if echo "$out" | grep -q "$m"; then
      scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:$BASE/model_outputs/${m}_outputs.json \
        $LOCAL/model_outputs/ 2>/dev/null && echo "PULLED $m at $(date)" && pulled=1
    fi
  done
  missing=0
  for m in "${MODELS[@]}"; do [ -s "$LOCAL/model_outputs/${m}_outputs.json" ] || missing=1; done
  if [ "$missing" == "0" ]; then echo "ALL 6 DONE $(date)"; exit 0; fi
  sleep 60
done
echo "POLL_END"