#!/bin/bash
# setup_pif.sh
# Run ONCE on login node after activating dlm_attack env.
# Sets up the PiF fork and drops in our adapter files.

set -e
PROJECT=/scratch/si2356/dlm-jailbreak-transfer
conda activate /scratch/si2356/conda-envs/dlm_attack

# ── 1. Clone repo ──────────────────────────────────────────────────────────
if [ ! -d "$PROJECT/PiF" ]; then
    echo "Cloning PiF repo..."
    cd $PROJECT
    git clone https://github.com/tmllab/2025_ICLR_PiF PiF
else
    echo "PiF repo already exists — pulling latest..."
    cd $PROJECT/PiF && git pull
fi

# ── 2. Drop in our adapter files ───────────────────────────────────────────
echo "Copying adapter files..."
cp pif_target_models.py $PROJECT/PiF/
cp run_pif.py           $PROJECT/PiF/
chmod +x $PROJECT/PiF/run_pif.py

# ── 3. Install any missing deps (BERT + sentence-transformers) ─────────────
echo "Checking dependencies..."
pip install sentence-transformers --quiet --break-system-packages 2>/dev/null || true

# ── 4. Download BERT-Large if not cached ───────────────────────────────────
python3 - << 'PYEOF'
import os
os.environ["HF_HOME"] = "/scratch/si2356/.cache/huggingface"
from huggingface_hub import snapshot_download
path = "/scratch/si2356/.cache/huggingface/hub"
try:
    snapshot_download("google-bert/bert-large-uncased", cache_dir=path,
                      ignore_patterns=["*.gguf","flax_*","tf_*"])
    print("BERT-Large: OK")
except Exception as e:
    print(f"BERT-Large download: {e} — will use offline cache if available")
PYEOF

# ── 5. Create output dirs ──────────────────────────────────────────────────
mkdir -p $PROJECT/PiF/logs
for model in qwen2.5 falcon llama llada dream diffucoder; do
    for dataset in advbench malicious_instruct harmbench jailbreakbench strongreject; do
        mkdir -p $PROJECT/results/pif/$model/$dataset
    done
done

echo ""
echo "=== PiF setup complete ==="
echo "Repo:    $PROJECT/PiF/"
echo "Adapter: $PROJECT/PiF/pif_target_models.py"
echo "Runner:  $PROJECT/PiF/run_pif.py"
echo ""
echo "To run (after Stage 1 completes for ArrAttack):"
echo "  cd $PROJECT/PiF"
echo "  sbatch run_pif_qwen2.5.sh"
echo "  sbatch run_pif_falcon.sh"
echo "  sbatch run_pif_llama.sh"
echo "  sbatch run_pif_llada.sh"
echo "  sbatch run_pif_dream.sh"
echo "  sbatch run_pif_diffucoder.sh"
