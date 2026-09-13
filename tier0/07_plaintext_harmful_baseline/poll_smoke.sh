#!/bin/bash
# Wait for T07 smoke job 17652715 to finish, show log.
for i in $(seq 1 80); do
  out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
    'squeue -u bc3194 2>/dev/null | grep -c t07_smok' 2>/dev/null)
  if [ "$out" == "0" ] || [ -z "$out" ]; then
    echo "smoke finished at $(date)"
    for j in 1 2 3; do
      res=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
        'cat /scratch/bc3194/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline/t07_smoke_*.log 2>/dev/null | tail -25' 2>/dev/null)
      if [ -n "$res" ]; then echo "$res"; break; fi; sleep 3
    done
    exit 0
  fi
  sleep 50
done
echo TIMEOUT