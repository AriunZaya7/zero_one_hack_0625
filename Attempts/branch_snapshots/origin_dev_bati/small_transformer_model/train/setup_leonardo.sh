#!/bin/bash
# setup_leonardo.sh
# ==================
# Run this ONCE after SSHing into Leonardo to set up your environment.
#
# Usage:
#   bash setup_leonardo.sh

echo "Setting up environment on Leonardo..."

# ── Load modules ──────────────────────────────────────────────────────────────
module load python/3.10
module load cuda/12.1

# ── Create virtual environment ────────────────────────────────────────────────
echo "Creating virtual environment..."
python -m venv ~/venv
source ~/venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── WandB login ───────────────────────────────────────────────────────────────
echo ""
echo "Logging into WandB..."
echo "Paste your WandB API key when prompted:"
wandb login

# ── Download model weights ahead of time ─────────────────────────────────────
# This downloads Qwen2-0.5B to the HuggingFace cache (~1GB)
# Much better to do this now than during the SLURM job
echo ""
echo "Pre-downloading Qwen2-0.5B model weights (~1GB)..."
python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Downloading tokenizer...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
print("Download complete!")
EOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo "  1. Upload your data folder:"
echo "     scp -r data/ USERNAME@login.leonardo.cineca.it:~/zero_one_hack_0625/small_transformer_model/"
echo ""
echo "  2. Run a quick sanity check (debug mode, no GPU needed):"
echo "     python train_model.py --debug"
echo ""
echo "  3. Submit the full training job:"
echo "     sbatch job.slurm"
echo ""
echo "  4. Monitor the job:"
echo "     squeue -u \$USER"
echo "     tail -f logs/train_\$(squeue -u \$USER -h -o %i).log"
echo "============================================"
