"""R2 Step 13 — split the 913-prompt pool into 10 shards (~92 each) for the two
slowest victims (dream, diffucoder) so they run in parallel on HPC.

Every victim shares the same 913 prompts; the shard split is at the GLOBAL pool
level (pool_913.csv order: harmbench 400, strongreject 313, jbb 100, mi 100).
Each shard k is a contiguous ~92-row (last shard 85) slice of the 913 pool, written
into 4 dataset files matching stage5's schema, so a stage5 run pointed at a shard
sees exactly that shard's prompts.

Output: step13_dataset_shard/<k> for k in 01..10, each with
  harmbench/text_all.csv, strongreject/strongreject.csv,
  jailbreakbench/jailbreakbench.csv, malicious_instruct.txt
(only files whose dataset has prompts in this shard; empty dirs otherwise.)
"""
import os
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = HERE.parent / "02_dedup_and_config_audit" / "pool_913.csv"
OUT = HERE / "step13_dataset_shard"

pool = pd.read_csv(POOL)
assert len(pool) == 913, len(pool)
# order within the file is already grouped by dataset (harmbench..mi), sequential idx
N = len(pool)
NSHARD = 10
# boundaries: first 9 shards get 92, last gets 85 (913 - 9*92 = 85)
sizes = [92] * 9 + [85]
bounds = []
start = 0
for s in sizes:
    bounds.append((start, start + s))
    start += s
assert start == 913, start
print("shard bounds:", bounds)

OUT.mkdir(exist_ok=True)
for k, (a, b) in enumerate(bounds, 1):
    sub = pool.iloc[a:b].reset_index(drop=True)
    kdir = OUT / f"{k:02d}"
    for ds in ["harmbench", "strongreject", "jailbreakbench", "malicious_instruct"]:
        d = sub[sub["dataset"] == ds]
        (kdir / ds).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Behavior": sub[sub.dataset == "harmbench"]["original_prompt"].tolist()}).to_csv(kdir / "harmbench" / "text_all.csv", index=False)
    pd.DataFrame({"forbidden_prompt": sub[sub.dataset == "strongreject"]["original_prompt"].tolist()}).to_csv(kdir / "strongreject" / "strongreject.csv", index=False)
    pd.DataFrame({"Goal": sub[sub.dataset == "jailbreakbench"]["original_prompt"].tolist()}).to_csv(kdir / "jailbreakbench" / "jailbreakbench.csv", index=False)
    with open(kdir / "malicious_instruct" / "malicious_instruct.txt", "w") as f:
        f.write("\n".join(sub[sub.dataset == "malicious_instruct"]["original_prompt"].tolist()) + "\n")
    n_in = len(sub)
    print(f"shard {k:02d}: rows={n_in}  "
          f"hb={int((sub.dataset=='harmbench').sum())} "
          f"sr={int((sub.dataset=='strongreject').sum())} "
          f"jb={int((sub.dataset=='jailbreakbench').sum())} "
          f"mi={int((sub.dataset=='malicious_instruct').sum())}")
print("wrote", OUT)