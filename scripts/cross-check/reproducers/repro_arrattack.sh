#!/usr/bin/env bash
#
# repro_arrattack.sh — Faithfully re-run ArrAttack for ONE model over ONLY the
# sampled (dataset,prompt_idx) block, regenerating target_response/final_response.
#
# Args: data_dir hf_cache model outdir prompt_idx prompt
#
# stage5_attack.py drives a multi-attempt loop (GPTFuzz-style) and writes
# arrattack_results.csv. To reproduce one decision we re-run the script but point
# its PROJECT_DIR/DATASET_BASE at a slim data dir containing ONLY the sampled
# prompt in its dataset file and the levels it expects. Because the loop is
# stochastic, exact byte equality is not expected; 06_compare.py judges agreement.
#
set -euo pipefail
DATA_DIR="${1:?data_dir}"; HF_CACHE="${2:?hf_cache}"; MODEL="${3:?model}"
OUTDIR="${4:?outdir}"; IDX="${5:?prompt_idx}"; PROMPT="${6:?prompt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$HERE/../stage5_attack.py"

mkdir -p "$OUTDIR"
[[ -f "$STAGE" ]] || { echo "stage5_attack.py not found: $STAGE"; exit 3; }

WORK="$(mktemp -d)/run"; mkdir -p "$WORK"
cp "$STAGE" "$WORK/stage5_attack.py"

# slim dataset dir: put sampled prompt as the ONLY row of harmbench (first in plan)
SLIM="$WORK/slim"; mkdir -p "$SLIM/dataset/harmbench"
# always keep a header; stage5 keys on these files
if [[ -f "$DATA_DIR/dataset/harmbench/text_all.csv" ]]; then
  python3 - "$DATA_DIR" "$PROMPT" "$SLIM" <<'PY'
import csv,sys,ast
data,prompt,slim=sys.argv[1],sys.argv[2],sys.argv[3]
src=f"{data}/dataset/harmbench/text_all.csv"
rows=list(csv.DictReader(open(src,newline='',encoding='utf-8')))
header=list(rows[0].keys()) if rows else ['Behavior']
row=next((r for r in rows if r.get('Behavior','').strip()==prompt.strip()),None)
if row is None:
    row={header[0]:prompt}
for k in header: row.setdefault(k,'')
with open(f"{slim}/dataset/harmbench/text_all.csv","w",newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=header)
    w.writeheader(); w.writerow(row)
PY
else
  echo "Behavior" > "$SLIM/dataset/harmbench/text_all.csv"
  echo "$PROMPT" >> "$SLIM/dataset/harmbench/text_all.csv"
fi

# patch stage5 path constants: PROJECT_DIR + ArrAttack sys.path to our slim/data
sed -i "s#/scratch/si2356/dlm-jailbreak-transfer#$SLIM#g" "$WORK/stage5_attack.py"
# sed the ArrAttack util import path too (harmless if dir absent -> script errors skipping)
mkdir -p "$SLIM/ArrAttack/utils"

export TARGET="$MODEL"
export TRANSFORMERS_OFFLINE=1
cd "$WORK"
echo "[repro-arrattack] $MODEL idx=$IDX"
if python3 "$WORK/stage5_attack.py" > "$OUTDIR/$IDX.log" 2>&1; then
  # collect rows
  cp -f arrattack_results.csv "$OUTDIR/" 2>/dev/null || true
  echo "REPRODUCED $MODEL (see $OUTDIR/arrattack_results.csv)"
  exit 0
else
  echo "repro-arrattack FAILED (see $OUTDIR/$IDX.log)"; exit 4
fi