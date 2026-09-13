"""
R2 Step 13 — build the FULL-POOL ArrAttack dataset files from pool_913.csv.

stage5_attack.py reads per-dataset files and takes the LAST N prompts
(DATASET_CONFIG last_n). For the full-pool run we rebuild every file containing
ALL 913 prompts in the pool order (harmbench 400 -> strongreject 313 ->
jailbreakbench 100 -> malicious_instruct 100), with the columns stage5 expects,
and set last_n = full count so no slicing happens.

Outputs (into ArrAttack/dataset/<ds>/):
  harmbench/text_all.csv        col "Behavior"
  strongreject/strongreject.csv col "forbidden_prompt"
  jailbreakbench/jailbreakbench.csv col "Goal"
  malicious_instruct/malicious_instruct.txt  one prompt per line
"""
import os, csv, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "..", "02_dedup_and_config_audit", "pool_913.csv")
# Always build into a local, writable dir (ArrAttack/dataset is a symlink to a
# student HPC path that is NOT writable/readable here).
ARR = os.path.join(HERE, "..", "..", "ArrAttack")
OUT_BASE = os.path.join(HERE, "step13_dataset")

pool = pd.read_csv(POOL)
print("pool total:", len(pool), pool["dataset"].value_counts().to_dict())

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(os.path.join(OUT_BASE, "harmbench"), exist_ok=True)
os.makedirs(os.path.join(OUT_BASE, "strongreject"), exist_ok=True)
os.makedirs(os.path.join(OUT_BASE, "jailbreakbench"), exist_ok=True)
os.makedirs(os.path.join(OUT_BASE, "malicious_instruct"), exist_ok=True)

hb = pool[pool.dataset == "harmbench"]["original_prompt"].tolist()
sr = pool[pool.dataset == "strongreject"]["original_prompt"].tolist()
jb = pool[pool.dataset == "jailbreakbench"]["original_prompt"].tolist()
mi = pool[pool.dataset == "malicious_instruct"]["original_prompt"].tolist()
print(f"harmbench={len(hb)} strongreject={len(sr)} jailbreakbench={len(jb)} malicious={len(mi)}")

pd.DataFrame({"Behavior": hb}).to_csv(
    os.path.join(OUT_BASE, "harmbench", "text_all.csv"), index=False)
pd.DataFrame({"forbidden_prompt": sr}).to_csv(
    os.path.join(OUT_BASE, "strongreject", "strongreject.csv"), index=False)
pd.DataFrame({"Goal": jb}).to_csv(
    os.path.join(OUT_BASE, "jailbreakbench", "jailbreakbench.csv"), index=False)
with open(os.path.join(OUT_BASE, "malicious_instruct", "malicious_instruct.txt"), "w") as f:
    f.write("\n".join(mi) + "\n")

# manifest of last_n (full counts) for the stage5 full-pool variant
manifest = {
    "harmbench": len(hb), "strongreject": len(sr),
    "jailbreakbench": len(jb), "malicious_instruct": len(mi),
    "out_base": OUT_BASE,
}
with open(os.path.join(HERE, "step13_dataset_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=4, ensure_ascii=False)

print("built full-pool dataset files at", OUT_BASE)
print("manifest ->", os.path.join(HERE, "step13_dataset_manifest.json"))