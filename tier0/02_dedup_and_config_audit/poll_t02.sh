#!/bin/bash
# poll HPC job 17642675 until response_length_stats.csv exists, then pull it back
for i in $(seq 1 120); do
  out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
    'ls /scratch/bc3194/dlm-jailbreak-transfer/tier0/02_dedup_and_config_audit/response_length_stats.csv 2>/dev/null && echo DONE' 2>/dev/null)
  if echo "$out" | grep -q DONE; then
    echo "JOB DONE at $(date)"
    scp -o ConnectTimeout=25 bc3194@jubail.abudhabi.nyu.edu:/scratch/bc3194/dlm-jailbreak-transfer/tier0/02_dedup_and_config_audit/response_length_stats.csv \
      /home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/02_dedup_and_config_audit/response_length_stats.csv 2>/dev/null && echo "PULLED BACK"
    echo "=== job tail ==="
    timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 bc3194@jubail.abudhabi.nyu.edu \
      'tail -25 /scratch/bc3194/dlm-jailbreak-transfer/t02_tokenlen_17642675.out 2>/dev/null; echo ===ERR===; tail -15 /scratch/bc3194/dlm-jailbreak-transfer/t02_tokenlen_17642675.err 2>/dev/null' 2>/dev/null
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT polling (job still running)"