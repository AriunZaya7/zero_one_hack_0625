#!/bin/bash
#SBATCH --job-name=gpt-procseq
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=01:30:00
#SBATCH --output=outputs/slurm/gpt_%j.log
#SBATCH --error=outputs/slurm/gpt_%j.log

echo "======================================"
echo "Job     : $SLURM_JOB_ID  (gpt-procseq)"
echo "Node    : $SLURMD_NODENAME"
echo "Start   : $(date)"
echo "======================================"

module load python/3.11.7
module load cuda/12.1
source ~/venv/bin/activate

cd ~/zero_one_hack_0625
mkdir -p outputs/slurm outputs/gpt_small results

export TOKENIZERS_PARALLELISM=false

echo ""
echo "=== [1/4] ID run — train on all 3 families ==="
python train.py \
    --model gpt:small \
    --train-families mosfet igbt ic \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_small_allfam

echo ""
echo "=== [2/4] LOO MOSFET ==="
python train.py \
    --model gpt:small \
    --leave-out mosfet \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_small_loo_mosfet

echo ""
echo "=== [3/4] LOO IGBT ==="
python train.py \
    --model gpt:small \
    --leave-out igbt \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_small_loo_igbt

echo ""
echo "=== [4/4] LOO IC ==="
python train.py \
    --model gpt:small \
    --leave-out ic \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_small_loo_ic

echo ""
echo "=== Running eval pipeline on all-fam checkpoint ==="
python eval_runner.py \
    --model gpt:small \
    --checkpoint outputs/gpt_small_allfam \
    --score

echo "======================================"
echo "Done: $(date)"
echo "======================================"
