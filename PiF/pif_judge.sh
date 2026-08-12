#!/bin/bash
#SBATCH --job-name=pif_judge
#SBATCH --output=/scratch/si2356/dlm-jailbreak-transfer/logs/pif_judge_%j.out
#SBATCH --error=/scratch/si2356/dlm-jailbreak-transfer/logs/pif_judge_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="/scratch/si2356/dlm-jailbreak-transfer"
PIF_DIR="$PROJECT/PiF"
RESULTS_BASE="$PROJECT/results/pif"
JUDGE_SCRIPT="$PIF_DIR/pif_judge.py"
HF_CACHE="/scratch/si2356/.cache/huggingface/hub"
LOG_DIR="$PROJECT/logs"

mkdir -p "$LOG_DIR"

# ── Copy latest judge script into place ───────────────────────────────────────
cp "$(dirname "$0")/pif_judge.py" "$JUDGE_SCRIPT"
echo "[setup] Copied pif_judge.py → $JUDGE_SCRIPT"

# ── Environment ───────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate base 2>/dev/null || true

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

echo "[env] Python: $(which python3)"
echo "[env] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# ── Pre-cache GPTFuzz model (runs fast if already cached) ─────────────────────
echo "[cache] Ensuring hubert233/GPTFuzz is cached..."
python3 - << 'PYEOF'
from transformers import RobertaTokenizer, RobertaForSequenceClassification
print("  Downloading/verifying tokenizer...")
RobertaTokenizer.from_pretrained("hubert233/GPTFuzz")
print("  Downloading/verifying model weights...")
RobertaForSequenceClassification.from_pretrained("hubert233/GPTFuzz")
print("  hubert233/GPTFuzz ready.")
PYEOF

# ── Targets to judge ──────────────────────────────────────────────────────────
# Add or remove targets as needed
TARGETS=(
    "qwen2.5"
    "falcon-h1"
    "llama3.2"
    "llada"
    "dream"
    "diffucoder"
)
DATASET="harmbench"

# ── Run judge for each target ─────────────────────────────────────────────────
OVERALL_PASS=0
OVERALL_SKIP=0
OVERALL_FAIL=0

for TARGET in "${TARGETS[@]}"; do
    RESULTS_CSV="$RESULTS_BASE/$TARGET/$DATASET/results.csv"
    OUTPUT_CSV="$RESULTS_BASE/$TARGET/$DATASET/results_judged.csv"

    if [[ ! -f "$RESULTS_CSV" ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[SKIP] $TARGET — results.csv not found at $RESULTS_CSV"
        (( OVERALL_SKIP++ )) || true
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START] target=$TARGET  dataset=$DATASET"
    echo "        input  → $RESULTS_CSV"
    echo "        output → $OUTPUT_CSV"

    python3 "$JUDGE_SCRIPT" \
        --results_csv "$RESULTS_CSV" \
        --output_csv  "$OUTPUT_CSV" \
    && (( OVERALL_PASS++ )) || (( OVERALL_FAIL++ )) || true

    echo "[DONE]  target=$TARGET"
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "ALL TARGETS COMPLETE"
echo "  passed : $OVERALL_PASS"
echo "  skipped: $OVERALL_SKIP"
echo "  failed : $OVERALL_FAIL"
echo ""

# Print per-target ASR summary from judged CSVs
echo "GPTFuzz ASR Summary:"
echo "────────────────────"
for TARGET in "${TARGETS[@]}"; do
    OUTPUT_CSV="$RESULTS_BASE/$TARGET/$DATASET/results_judged.csv"
    if [[ -f "$OUTPUT_CSV" ]]; then
        python3 - "$TARGET" "$OUTPUT_CSV" << 'PYEOF'
import sys, csv
target, path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path, encoding="utf-8")))
n    = len(rows)
hits = sum(1 for r in rows if str(r.get("judge_gptfuzz","")).strip() == "1")
print(f"  {target:<16} {hits:>3}/{n:<3} = {hits/n*100:5.1f}%  [{path}]")
PYEOF
    fi
done
echo "════════════════════════════════════════════════════════════"
