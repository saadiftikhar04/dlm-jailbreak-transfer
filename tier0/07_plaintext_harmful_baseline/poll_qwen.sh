#!/bin/bash
# Poll T07 qwen generation job(s); pull outputs when the qwen outputs exist.
MODEL_KEY=qwen2.5
HOST=bc3194@jubail.abudhabi.nyu.edu
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
for i in $(seq 1 100); do
  out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 $HOST \
    "ls $BASE/model_outputs/qwen_outputs.json 2>/dev/null && echo DONE" 2>/dev/null)
  if echo "$out" | grep -q DONE; then
    echo "QWEN DONE at $(date)"
    scp -o ConnectTimeout=25 $HOST:$BASE/model_outputs/qwen_outputs.json \
      /home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline/model_outputs/ 2>/dev/null && echo PULLED
    timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 $HOST \
      "tail -8 $BASE/model_outputs/qwen_outputs.json | head -c 400" 2>/dev/null
    exit 0
  fi
  # also report if job died
  st=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 $HOST \
    'squeue -u bc3194 2>/dev/null | grep -c t07_gen' 2>/dev/null)
  sleep 60
done
echo "TIMEOUT"