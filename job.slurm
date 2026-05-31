#!/bin/bash
#SBATCH --job-name=kremsians-industrial-ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

mkdir -p outputs/slurm

python -m pip install -r requirements.txt
python generate_submission_2_families.py --count-per-family 200
python train.py \
  --model gpt:large \
  --submission-2-ood \
  --epochs 30 \
  --batch-size 64 \
  --out outputs/gpt_large_train12
python eval_guided.py \
  --kind gpt \
  --size large \
  --submission-2-ood \
  --checkpoint outputs/gpt_large_train12 \
  --eval-seqs 500
python eval_runner.py \
  --model gpt:large \
  --checkpoint outputs/gpt_large_train12 \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out final_submission/SUBMISSION_2_gpt_large_official \
  --official-names
