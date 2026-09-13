#!/bin/bash
# Poll all 6 T07 victim generation jobs; pull each completed output.
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
LOCAL=/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline
MODELS=(qwen llama falcon llada dream diffucoder)
for round in $(seq 1 200); do
  pulled=0
  for m in "${MODELS[@]}"; do
    # local already has it? skip
    if [ -f "$LOCAL/model_outputs/${m}_outputs.json" ]; then continue; fi
    out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
      "ls $BASE/model_outputs/${m}_outputs.json 2>/dev/null && echo DONE" 2>/dev/null)
    if echo "$out" | grep -q DONE; then
      scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:$BASE/model_outputs/${m}_outputs.json \
        $LOCAL/model_outputs/ 2>/dev/null && echo "PULLED ${m} at $(date)" && pulled=1
    fi
  done
  done_left=0
  for m in "${MODELS[@]}"; do [ -f "$LOCAL/model_outputs/${m}_outputs.json" ] && done_left=1; done
  if [ "$done_left" == "0" ] && [ "$pulled" == "0" ]; then
    # check if jobs still running
    nrun=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
      'squeue -u bc3194 2>/dev/null | grep -c t07_gen' 2>/dev/null)
    [ "$nrun" == "0" ] && [ "$pulled" == "0" ] && echo "all done or failed, no outputs" && break
  fi
  sleep 90
done
echo "POLL_END"