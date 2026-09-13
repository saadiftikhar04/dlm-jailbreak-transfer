#!/bin/bash
# Re-poll: pull only the two still-incomplete files (diffucoder_seed3, llada_seed3)
# once their HPC line count reaches target. diffucoder target 30, llada target 30.
BASE=/scratch/bc3194/dlm-jailbreak-transfer/tier0/12_low_asr_multiseed/model_outputs
LOCAL=/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/12_low_asr_multiseed/model_outputs
for piece in "diffucoder_seed3 30" "llada_seed3 30"; do
  set -- $piece; f=$1; tgt=$2
  for round in $(seq 1 120); do
    [ -s "$LOCAL/$f" ] && [ "$(wc -l < $LOCAL/$f)" -ge "$tgt" ] && break
    n=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
        "wc -l < $BASE/$f 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$n" ] && [ "$n" -ge "$tgt" ] 2>/dev/null; then
      scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:$BASE/$f $LOCAL/ 2>/dev/null && \
        { echo "PULLED $f ($n/$tgt) $(date)"; break; }
    fi
    sleep 60
  done
done
echo "DONE $(date)"
echo "local complete: $(ls $LOCAL/*_seed*_outputs.json 2>/dev/null | wc -l) files"
for f in $LOCAL/*_seed*_outputs.json; do [ -f "$f" ] && echo "$(basename $f): $(wc -l < $f)"; done