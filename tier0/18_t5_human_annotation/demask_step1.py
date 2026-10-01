"""
STEP 1: Demask MetaCipher final_responses ([MASKn] -> detected_keywords[i], matching the
official judge's own mapping in qwen_metacipher_judge.py / qwen_metacipher_inference.py).

Non-destructive: writes NEW files, never modifies originals.
 - demasked metacipher judged CSVs -> results/metacipher/Metacipher_Judged_demasked/{model}.csv
 - rebuilt all_final_responses.jsonl  -> tier0/11_common_judge_rescore/all_final_responses_demasked.jsonl
 - updated T5 annotation sample      -> tier0/18_t5_human_annotation/t5_annotation_sample_demasked.csv
   (all 230 rows, response demasked for metacipher rows, carries over the 46 human_label values)
"""
import pandas as pd, json, os, sys, re, collections

A='/home/bc3194/Desktop/dlm-jailbreak-transfer/'
MODELS=["qwen","llama","falcon","llada","dream","diffucoder"]

def demask(resp, keyw_col):
    """Official judge mapping: [MASK{i+1}] -> keyw[i] for i,kw in enumerate(keyw)."""
    resp=str(resp)
    if not keyw_col or (isinstance(keyw_col,float) and pd.isna(keyw_col)):
        return resp, 0
    if isinstance(keyw_col,str):
        try: kws=eval(keyw_col)
        except Exception: kws=[]
    else:
        kws=list(keyw_col)
    if not isinstance(kws,list): kws=[]
    n=0
    for i,kw in enumerate(kws):
        m=f"[MASK{i+1}]"
        c=resp.count(m)
        if c: resp=resp.replace(m,str(kw)); n+=c
    return resp, n

# ---------- 1. demasked judged CSVs ----------
outdir=A+"results/metacipher/Metacipher_Judged_demasked"
os.makedirs(outdir, exist_ok=True)
summary={}
for m in MODELS:
    df=pd.read_csv(f"{A}results/metacipher/Metacipher_Judged/{m}.csv")
    before=int(df['final_response'].astype(str).str.contains(r'\[MASK\d+\]', regex=True).sum())
    reps=0; changed=0
    rows=[]
    for _,r in df.iterrows():
        d,nrep=demask(r['final_response'], r.get('detected_keywords'))
        reps+=nrep
        if nrep>0: changed+=1
        r=r.copy()
        r['final_response']=d
        rows.append(r)
    ndf=pd.DataFrame(rows)
    ndf.to_csv(f"{outdir}/{m}.csv", index=False)
    after=int(ndf['final_response'].astype(str).str.contains(r'\[MASK\d+\]', regex=True).sum())
    summary[m]=(before,after,reps,changed)
    print(f"{m:11s} rows_with_MASK before={before} after={after}  total_replacements={reps} rows_changed={changed}")
print("")

# ---------- 2. rebuild all_final_responses.jsonl (demasked metacipher; pif/arrattack unchanged) ----------
# Mirror extract_all_responses.py exactly but read metacipher from the demasked dir.
def load_df(attack, model):
    if attack=="pif":  return pd.read_csv(f"{A}results/pif/PIF_JUDGED/{model}_pif_final_judged.csv")
    if attack=="metacipher": return pd.read_csv(f"{outdir}/{model}.csv")
    if attack=="arrattack": return pd.read_csv(f"{A}results/arrattack/Arrattack_Judged/arrattack_{model}_judged.csv")

DEF={
 "pif":{"orig":"original_prompt","attacked":"pif_prompt","resp":"victim_output","label":"llm_judge"},
 "metacipher":{"orig":"original_prompt","attacked":"final_converted_prompt","resp":"final_response","label":"llm_judge"},
 "arrattack":{"orig":"original_prompt","attacked":"final_converted_prompt","resp":"final_response","label":"gpt_fuzz"},
}
def _to_native(o):
    import math
    if hasattr(o, "item"):
        return _to_native(o.item())
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(x) for x in o]
    return o

rows=[]; n=0
for attack,models in [("pif",MODELS),("metacipher",MODELS),("arrattack",MODELS)]:
    d=DEF[attack]
    for model in MODELS:
        df=load_df(attack,model)
        succ=(df["asr_success"] if attack!="pif" else pd.to_numeric(df[d["label"]],errors="coerce").fillna(0).astype(int)==1)
        for i in range(len(df)):
            r=df.iloc[i]
            rows.append({
                "attack":attack,"model":model,
                "model_family":"causal" if model in ("qwen","llama","falcon") else "diffusion",
                "dataset":r.get("dataset",""),"prompt_idx":int(r.get("prompt_idx",-1)) if pd.notna(r.get("prompt_idx")) else -1,
                "original_prompt":str(r.get(d["orig"],"")),"attacked_prompt":str(r.get(d["attacked"],"")),
                "final_response":str(r.get(d["resp"],"")),
                "official_judge_label":r.get(d["label"],""),
                "official_asr_success":bool(succ.iloc[i]),
                "source_file":(f"{A}results/metacipher/Metacipher_Judged_demasked/{model}.csv" if attack=="metacipher" else f"<original>"),
            }); n+=1
assert n==11946, f"expected 11946 got {n}"
outj=A+"tier0/11_common_judge_rescore/all_final_responses_demasked.jsonl"
with open(outj,"w") as f:
    for r in rows: f.write(json.dumps(_to_native(r),ensure_ascii=False)+"\n")
print(f"wrote {n} rows -> {outj}")
print("")

# ---------- 3. updated T5 annotation sample (demasked responses, labels carried over) ----------
samp=pd.read_csv(A+"tier0/18_t5_human_annotation/t5_annotation_sample.csv", dtype={'human_label':str})
# build keyword lookup per (model,dataset,prompt_idx) from demasked metacipher dirs
kwmap={}
for m in MODELS:
    df=pd.read_csv(f"{outdir}/{m}.csv")
    for _,r in df.iterrows():
        kwmap[("metacipher",m,str(r['dataset']),int(r['prompt_idx']))]=r['detected_keywords']

out=[]
for _,r in samp.iterrows():
    rr=r.to_dict()
    resp=str(rr['response'])
    kk=("metacipher"==rr['attack'])
    if kk:
        key=("metacipher",rr['model'],str(rr['dataset']),int(rr['prompt_idx']))
        kw=kwmap.get(key)
        if kw is not None:
            resp,_=demask(resp, kw)
        else:
            print(f"WARN no keyword lookup for {key}")
    rr['response']=resp
    out.append(rr)
ndf=pd.DataFrame(out)
cols=['id','overlap','attack','model','dataset','prompt_idx','original_prompt','response','human_label']
ndf=ndf[cols]
outs=A+"tier0/18_t5_human_annotation/t5_annotation_sample_demasked.csv"
ndf.to_csv(outs,index=False)
lab_filled=ndf['human_label'].notna() & (ndf['human_label'].astype(str).str.strip()!='')
print(f"wrote T5 sample -> {outs}")
print(f"  rows={len(ndf)}  labels_carried={lab_filled.sum()}  remaining_blank={(~lab_filled).sum()}")
mask_left=ndf['response'].astype(str).str.contains(r'\[MASK\d+\]',regex=True).sum()
print(f"  rows still containing [MASK] in response (should be 0 for metacipher): {mask_left}")
sys.stdout.write("ALL DONE STEP 1\n")