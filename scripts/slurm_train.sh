#!/bin/bash
#SBATCH --job-name=procseq
#SBATCH --partition=boost_usr_prod
#SBATCH --account=YOUR_ACCOUNT_HERE     # fill in (same one used in small_transformer_model/train/job.slurm)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.out

# ---------------------------------------------------------------------------
# Leonardo training launcher. Pass model + mode as args:
#   sbatch scripts/slurm_train.sh "hf:Qwen/Qwen2.5-0.5B" "--leave-out mosfet" 3
#   sbatch scripts/slurm_train.sh "gpt:small"            "--train-families mosfet igbt ic" 10
# Args: 1=MODEL  2=MODE_FLAGS  3=EPOCHS
#
# Modules match the teammate setup in
# small_transformer_model/train/{setup_leonardo.sh,job.slurm}. That job.slurm leaves
# --account as a placeholder too: fill in your real CINECA account before submitting.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${1:-gpt:tiny}"
MODE="${2:---leave-out mosfet}"
EPOCHS="${3:-3}"

module load python/3.10
module load cuda/12.1

# Activate the venv. The teammate setup uses ~/venv; create yours once with:
#   python -m venv ~/venv && source ~/venv/bin/activate
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install transformers scikit-learn matplotlib pandas wandb huggingface_hub
source ~/venv/bin/activate

# Offline mode (compute nodes are air-gapped). Pre-stage weights on the LOGIN node:
#   export HF_HOME=$WORK/hf_cache ; python scripts/prefetch_models.py
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline           # wandb sync from the login node afterwards

mkdir -p outputs/slurm

echo "MODEL=$MODEL  MODE=$MODE  EPOCHS=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun python train.py --model "$MODEL" $MODE --epochs "$EPOCHS"
