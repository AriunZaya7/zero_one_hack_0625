#!/bin/bash
#SBATCH --job-name=qwen-0.5b
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen-0.5b_%j.log
#SBATCH --error=logs/qwen-0.5b_%j.err

echo "=========================================="
echo "Job       : $SLURM_JOB_ID  (qwen-0.5b)"
echo "Node      : $SLURMD_NODENAME"
echo "Start     : $(date)"
echo "=========================================="

module load python/3.11.7
module load cuda/12.1

source ~/venv/bin/activate

cd ~/zero_one_hack_0625/small_transformer_model/train
mkdir -p logs metrics

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python train_model.py \
    --data_dir    ../data \
    --output_dir  $SCRATCH/checkpoints/qwen2.5-0.5b \
    --model_name  Qwen/Qwen2.5-0.5B \
    --epochs      10 \
    --batch_size  4 \
    --grad_accum  4 \
    --lr          1e-5 \
    --max_length  512 \
    --eval_every  50 \
    --save_every  500 \
    --run_name    qwen2.5-0.5b-run2

echo "Done: $(date)"
