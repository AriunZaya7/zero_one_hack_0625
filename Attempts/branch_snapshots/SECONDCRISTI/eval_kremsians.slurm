#!/bin/bash
#SBATCH --job-name=kremsians-eval
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/kremsians_%j.log
#SBATCH --error=outputs/slurm/kremsians_%j.log

echo "=========================================="
echo "Job : $SLURM_JOB_ID  (KREMSIANS eval)"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

module load python/3.11.7
module load cuda/12.1
source ~/venv/bin/activate
cd ~/zero_one_hack_0625
mkdir -p outputs/slurm
export TOKENIZERS_PARALLELISM=false

echo ""
echo "=== KREMSIANS OOD eval — GPT:small (fast) ==="
python eval_guided.py --kind gpt --size small --leave-out kremsians --eval-seqs 500

echo ""
echo "=== KREMSIANS OOD eval — GPT:large ==="
python eval_guided.py --kind gpt --size large --leave-out kremsians --eval-seqs 500

echo ""
echo "=== Submission pipeline on KREMSIANS ==="
python eval_runner.py \
    --model gpt:large \
    --checkpoint outputs/gpt_large_allfam \
    --valid  self_eval/eval_input_valid.csv \
    --anomaly self_eval/eval_input_anomaly.csv \
    --out submissions_kremsians/ \
    --score

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
