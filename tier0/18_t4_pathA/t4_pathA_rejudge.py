"""T4 (path A) re-judge of stored full-pool ArrAttack responses with the strict
judge. Universal key = normalized original_prompt (sharded victims renumber
global_idx, so gidx is unusable). Drops non-numeric global_idx rows (header
leakage from concatenated shards). Migrates any old "m|<gidx>" cache keys to
"m|<norm_prompt>" so already-judged llama/qwen/llada rows are not re-called.
"""
import os, json, time, re, collections
import pandas as pd
from openai import OpenAI

HERE = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/18_t4_pathA"
CACHE = f"{HERE}/t4_pathA_cache.jsonl"
syspath = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/18_t1_plaintext_control"
import sys
sys.path.insert(0, syspath)
import t1_judge  # noqa: F401

MODS = {"llama": "llama_results.csv", "qwen2.5": "qwen2.5_results.csv",
        "llada": "llada_results.csv", "dream": "dream_all_results.csv",
        "diffucoder": "diffucoder_all_results.csv", "falcon": "falcon_all_results.csv"}
IN = "/tmp/arrdiag"


def norm(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.strip('"\'“”‘’')
    return s


def read_clean(f):
    df = pd.read_csv(f, low_memory=False)
    # drop header leakage rows from concatenated shard files
    df = df[pd.to_numeric(df["global_idx"], errors="coerce").notna()]
    return df


def final_responses(df):
    df = df[df["target_response"].notna() & df["target_response"].astype(str).str.strip().ne("")].copy()
    df["_n"] = df["original_prompt"].map(norm)
    best = df.loc[df.groupby("_n")["llm_judge_score"].idxmax()].drop_duplicates("_n")
    return {r["_n"]: r for r in best.to_dict("records")}


def main():
    dsk = os.environ.get("DEEPSEEK_API_KEY", "")
    ork = os.environ.get("OPENROUTER_API_KEY", "")
    ds = OpenAI(api_key=dsk, base_url="https://api.deepseek.com")
    orc = OpenAI(api_key=ork, base_url="https://openrouter.ai/api/v1") if ork else None

    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["key"]] = r
    print(f"cache entries loaded: {len(done)}", flush=True)

    # Build gidx->norm map to migrate old keys for llama/qwen/llada
    gidx2norm = {}
    for m, f in MODS.items():
        p = os.path.join(IN, f)
        if os.path.exists(p):
            df = read_clean(p)
            for r in df.drop_duplicates("global_idx").to_dict("records"):
                gidx2norm[(m, int(r["global_idx"]))] = norm(r["original_prompt"])
    migrated = 0
    newdone = {}
    for k, r in done.items():
        mm, _, rest = k.partition("|")
        if rest.isdigit() and (mm, int(rest)) in gidx2norm:
            nk = f"{mm}|{gidx2norm[(mm, int(rest))]}"
            rr = dict(r)
            rr["key"] = nk
            if nk not in newdone:
                newdone[nk] = rr
                migrated += 1
        else:
            newdone[k] = r
    done = newdone
    print(f"cache after migration: {len(done)} (migrated {migrated})", flush=True)

    # Prepare final responses per model
    per = {}
    for m, f in MODS.items():
        p = os.path.join(IN, f)
        if os.path.exists(p):
            per[m] = final_responses(read_clean(p))
            print(f"{m}: final responses = {len(per[m])}", flush=True)

    out = open(f"{HERE}/t4_pathA_results.jsonl", "a")
    for m, mp in per.items():
        for nkey, r in mp.items():
            key = f"{m}|{nkey}"
            if key in done:
                continue
            resp = r["target_response"] or ""
            user = t1_judge.PIF_BINARY.format(prompt=(r["original_prompt"] or "")[:600], response=resp[:2400])
            row = {"key": key, "model": m, "dataset": r["dataset"], "original_prompt": r["original_prompt"],
                   "resp_len": len(resp), "sel_score": int(r["llm_judge_score"])}
            for _ in range(6):
                try:
                    row["strict_ds"] = t1_judge.parse_binary(t1_judge.call(ds, "deepseek-chat", t1_judge.PIF_BINARY, user))
                    break
                except Exception:
                    time.sleep(2)
            if orc:
                for _ in range(6):
                    try:
                        row["gpt_sol"] = t1_judge.parse_binary(t1_judge.call(orc, "openai/gpt-5.6-sol", t1_judge.PIF_BINARY, user))
                        break
                    except Exception:
                        time.sleep(2)
            else:
                row["gpt_sol"] = None
            blob = json.dumps(row, ensure_ascii=False)
            with open(CACHE, "a") as f:
                f.write(blob + "\n")
            out.write(blob + "\n")
            done[key] = row
            if len(done) % 200 == 0:
                print(f"  judged {len(done)}", flush=True)
    out.close()

    by = collections.defaultdict(list)
    for r in done.values():
        by[r["model"]].append(r)
    print("\nFULL-POOL STRICT ASR (fresh strict judge):")
    for m, rs in sorted(by.items()):
        n = len(rs)
        k = sum(1 for r in rs if r.get("strict_ds") == 1)
        kg = sum(1 for r in rs if r.get("gpt_sol") == 1)
        print(f"  {m:12s} n={n} strictDS={k/n*100:.1f}%  gpt_sol={kg/n*100:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupt")