<?xml version="1.0" encoding="UTF-8"?>
R2 Step 13 — migrate a subset of PENDING ArrAttack shards from the saturated
nvidia pool to the ebrainccs condo (A100-80G, MaxTRES gpu=4).

nvidia-xxl QoS is NOT attachable (Invalid qos specification since ~2026-08-13),
and the default nvidia pool is at its 12-job/user cap (QOSMaxJobsPerUserLimit),
so ebrainccs is the only pool with free capacity. We move 4 diffucoder shards.

Per-shard logic (idempotent / resume-safe: each shard's results live in
results_shard/<victim>_<shard>/ keyed by progress CSV, so a re-submit on a
different pool just continues):
  1. scancel any stale PENDING copy of this shard in the nvidia pool (by name).
  2. sbatch the ebrainccs variant.
EBRAINCCS_VARIANT_CLI is used so we do not edit sbatch files; but the template
already carries --qos=nvidia, so we generate an ebrainccs-edited copy on the node.
"""

import subprocess, time, sys

ROOT = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"
SYNC_HOST = None  # run locally, ssh to node

def sh(bash_cmd, host):
    r = subprocess.run(["ssh", "-o","ConnectTimeout=30", "-o","BatchMode=yes",
                        "bc3194@jubail.abudhabi.nyu.edu", bash_cmd],
                       capture_output=True, text=True)
    return r.stdout.strip(), r.returncode

SHARDS = ["01","02","03","04"]   # 4 diffucoder shards to ebrainccs (max 4 GPUs)

for s in SHARDS:
    name = f"arr_s{s}_diffucoder"
    # 1. scancel stale pending copies of this shard name in nvidia pool
    out, rc = sh(f"scancel --name={name} 2>&1; echo RC=$?", None)
    if rc == 0:
        time.sleep(2)

    # 2. build an ebrainccs variant of the shard sbatch (swap --qos=nvidia -> --qos=ebrainccs,
    #    -C 80g already ok on A100, -p nvidia -> -p condo for ebrainccs pool)
    src = f"{ROOT}/arr_shard_{s}_diffucoder.sbatch"
    dst = f"{ROOT}/arr_ebrain_{s}_diffucoder.sbatch"
    bash = (
        f"cd {ROOT} && "
        f"sed -e 's/--qos=nvidia/--qos=ebrainccs/g' "
        f"-e 's/--partition=nvidia/--partition=condo/g' "
        f"-e 's/--time=71:59:59/--time=71:59:59/g' "
        f"{src} > {dst} && "
        f"grep -E 'qos|partition' {dst} | head -4"
    )
    out, rc = sh(bash, None)
    print(f"[shard {s}] variant:\n{out}")

print("done generating ebrainccs variants")