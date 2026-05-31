#!/bin/bash
#SBATCH --job-name=qwen_finetune
#SBATCH --account=YOUR_ACCOUNT_HERE          # e.g. IscrXXXX_XXXXXXX — fill this in
#SBATCH --partition=boost_usr_prod           # GPU partition on Leonardo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                         # 1 GPU — increase to 4 if quota allows
#SBATCH --mem=32G
#SBATCH --time=02:00:00                      # 2 hours — adjust if needed
#SBATCH --output=logs/train_%j.log           # %j = job ID
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL                 # email when done or failed
#SBATCH --mail-user=YOUR_EMAIL_HERE          # fill this in

# ── Environment ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "=========================================="

# Load modules available on Leonardo
module load python/3.10
module load cuda/12.1

# Activate your virtual environment
source ~/venv/bin/activate

# Go to your project folder — update this path
cd ~/zero_one_hack_0625/small_transformer_model

# Create logs folder if it doesn't exist
mkdir -p logs

# ── WandB Login ───────────────────────────────────────────────────────────────
# Option A: set API key as env variable (recommended — no interactive prompt)
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE

# Option B: if you already ran wandb login on the cluster, comment out above
# and uncomment below:
# export WANDB_MODE=online

# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting training..."

python train_model.py \
    --data_dir      ./data \
    --output_dir    ./checkpoints \
    --model_name    Qwen/Qwen2-0.5B \
    --epochs        10 \
    --batch_size    4 \
    --grad_accum    4 \
    --lr            2e-4 \
    --max_length    512 \
    --eval_every    100 \
    --save_every    500 \
    --wandb_project industrial-ai-hackathon \
    --run_name      qwen2-0.5b-full-run

echo "=========================================="
echo "Done at: $(date)"
echo "=========================================="
