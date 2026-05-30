#!/bin/bash
# setup_leonardo.sh
# ==================
# Run this ONCE after SSHing into Leonardo to set up your environment.
#
# Usage:
#   bash setup_leonardo.sh

echo "Setting up environment on Leonardo..."

# ── Load modules ──────────────────────────────────────────────────────────────
module load python/3.11.7
module load cuda/12.1

# ── Create virtual environment ────────────────────────────────────────────────
echo "Creating virtual environment..."
python -m venv ~/venv
source ~/venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── Pre-download model weights to $SCRATCH (fast NVMe, no quota issues) ───────
echo ""
echo "Pre-downloading Qwen2.5 model weights to \$SCRATCH/hf_cache ..."
echo "(This may take a few minutes — run on login node which has internet)"

export HF_HOME=$SCRATCH/hf_cache

python - << 'EOF'
import os
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]

for m in MODELS:
    print(f"  Downloading {m} ...", flush=True)
    snapshot_download(repo_id=m)
    print(f"  Done: {m}", flush=True)

print("All models downloaded.")
EOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo "  1. Upload your data folder (if not already present):"
echo "     scp -r data/ USERNAME@login.leonardo.cineca.it:~/zero_one_hack_0625/small_transformer_model/"
echo ""
echo "  2. Run a quick sanity check (debug mode, no GPU needed):"
echo "     HF_HOME=\$SCRATCH/hf_cache python train_model.py --debug"
echo ""
echo "  3. Submit the full training job:"
echo "     sbatch job.slurm"
echo ""
echo "  4. Monitor the job:"
echo "     squeue -u \$USER"
echo "     tail -f logs/train_\$(squeue -u \$USER -h -o %i).log"
echo ""
echo "  5. View training dashboard:"
echo "     streamlit run dashboard.py"
echo "============================================"
