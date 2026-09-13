#!/bin/bash
cd /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool || exit 1
if pgrep -f "hpc_download_deps.py" >/dev/null 2>&1; then
  echo "already running"
  exit 0
fi
setsid nohup /scratch/bc3194/conda_envs/envs/raven/bin/python hpc_download_deps.py > hpc_download_deps.log 2>&1 < /dev/null &
echo "started PID $!"