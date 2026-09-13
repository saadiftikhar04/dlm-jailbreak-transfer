#!/bin/bash
# Monitor qwen T07 job 17652204 until qwen_outputs.json exists, then pull it.
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
LOCAL=/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
for i in $(seq 1 80); do
  out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
    "ls $BASE/model_outputs/qwen_outputs.json 2>/dev/null && echo DONE" 2>/dev/null)
  if echo "$out" | grep -q DONE; then
    echo "QWEN DONE at $(date)"
    scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:$BASE/model_outputs/qwen_outputs.json $LOCAL/model_outputs/ 2>/dev/null && echo PULLED
    timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
      "cd $BASE && tail -6 t07_gen_17652204.out 2>/dev/null" 2>/dev/null
    exit 0
  fi
  sleep 45
done
echo TIMEOUT