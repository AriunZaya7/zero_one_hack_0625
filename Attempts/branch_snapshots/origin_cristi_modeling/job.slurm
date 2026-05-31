#!/bin/bash
#SBATCH --job-name=infineon-submit
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=02:30:00
#SBATCH --output=outputs/slurm/submit_%j.log
#SBATCH --error=outputs/slurm/submit_%j.log

# ============================================================================
#  Industrial AI / Infineon hackathon — full REPRODUCIBLE submission pipeline.
#
#  One job, run on a Leonardo A100, that:
#    1. Rebuilds the self-eval split (ground truth for honest local metrics)
#    2. Retrains the scaling ladder from scratch on GPU (tiny/small/large),
#       saving per-epoch loss curves
#    3. Generates the OFFICIAL-format submission CSVs (valid_0001 IDs) from the
#       organiser input files in participant_files/
#    4. Generates self-eval predictions and scores them with the OFFICIAL
#       scorer (Task 1 Top-k/MRR, Task 2 NED/Exact/Token/Block, Task 3 F1/AUC)
#    5. Runs the KREMSIANS 4th-family OOD probe + model-size-vs-OOD sweep
#
#  Submit with:  sbatch job.slurm
# ============================================================================

echo "======================================================"
echo "Job   : $SLURM_JOB_ID  (infineon-submit)"
echo "Node  : $SLURMD_NODENAME"
echo "Start : $(date)"
echo "======================================================"

module load python/3.11.7
module load cuda/12.1
source ~/venv/bin/activate

cd ~/zero_one_hack_0625
mkdir -p outputs/slurm results submissions submissions_official

export TOKENIZERS_PARALLELISM=false

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "=== Dataset sizes ==="
python3 - << 'EOF'
from data import load_family, FAMILIES
for fam in FAMILIES:
    print(f"  {fam}: {len(load_family(fam))} sequences")
EOF

# ── STEP 0: (re)build the self-eval split with injected violations ──────────
echo ""
echo "=== [0/6] Build self-eval split (ground truth for local metrics) ==="
python make_selfeval.py

# ── STEP 1: Retrain the scaling ladder from scratch (loss curves saved) ─────
echo ""
echo "=== [1/6] Train GPT:tiny — all 3 real families (mosfet igbt ic) ==="
python train.py --model gpt:tiny  --train-families mosfet igbt ic \
    --epochs 20 --batch-size 64 --lr 3e-4 --out outputs/gpt_tiny_allfam

echo ""
echo "=== [1/6] Train GPT:small — all 3 real families ==="
python train.py --model gpt:small --train-families mosfet igbt ic \
    --epochs 20 --batch-size 64 --lr 3e-4 --out outputs/gpt_small_allfam

echo ""
echo "=== [1/6] Train GPT:large — all 3 real families  (SUBMISSION MODEL) ==="
python train.py --model gpt:large --train-families mosfet igbt ic \
    --epochs 30 --batch-size 64 --lr 3e-4 --out outputs/gpt_large_allfam

# ── STEP 2: OFFICIAL submission CSVs (correct valid_0001 / anomaly IDs) ─────
echo ""
echo "=== [2/6] Generate OFFICIAL submission CSVs (gpt:large) ==="
python eval_runner.py \
    --model gpt:large --checkpoint outputs/gpt_large_allfam \
    --valid   participant_files/eval_input_valid.csv \
    --anomaly participant_files/eval_input_anomaly.csv \
    --out submissions_official

# ── STEP 3: Self-eval predictions (we hold ground truth here) ───────────────
echo ""
echo "=== [3/6] Self-eval predictions — gpt:large ==="
python eval_runner.py \
    --model gpt:large --checkpoint outputs/gpt_large_allfam \
    --valid   self_eval/eval_input_valid.csv \
    --anomaly self_eval/eval_input_anomaly.csv \
    --out submissions --score

echo ""
echo "=== [3/6] Self-eval predictions — n-gram baseline ==="
python eval_runner.py --model ngram \
    --valid   self_eval/eval_input_valid.csv \
    --anomaly self_eval/eval_input_anomaly.csv \
    --out submissions --score

# ── STEP 4: Score self-eval with the OFFICIAL metric code ───────────────────
echo ""
echo "=== [4/6] OFFICIAL-scorer self-eval metrics — gpt:large ==="
python score_selfeval.py --tag gpt_large_ckpt --sub-dir submissions

echo ""
echo "=== [4/6] OFFICIAL-scorer self-eval metrics — n-gram baseline ==="
python score_selfeval.py --tag ngram --sub-dir submissions

# ── STEP 5: KREMSIANS 4th-family OOD + model-size sweep ─────────────────────
echo ""
echo "=== [5/6] KREMSIANS OOD — model-size sweep (tiny/small/large) ==="
python eval_guided.py --kind gpt --size tiny  --leave-out kremsians --eval-seqs 500
python eval_guided.py --kind gpt --size small --leave-out kremsians --eval-seqs 500
python eval_guided.py --kind gpt --size large --leave-out kremsians --eval-seqs 500

# ── STEP 6: Sanity-check the official submission shape ──────────────────────
echo ""
echo "=== [6/6] Submission sanity check ==="
python3 - << 'EOF'
import csv
from pathlib import Path
for name, cols in [("task1_gpt_large_ckpt.csv", ["EXAMPLE_ID","RANK_1","RANK_2","RANK_3","RANK_4","RANK_5"]),
                   ("task2_gpt_large_ckpt.csv", ["EXAMPLE_ID","PREDICTED_SEQUENCE"]),
                   ("task3_gpt_large_ckpt.csv", ["EXAMPLE_ID","IS_VALID","SCORE","PREDICTED_RULE"])]:
    p = Path("submissions_official")/name
    rows = list(csv.DictReader(p.open()))
    print(f"  {name}: {len(rows)} rows, header={list(rows[0].keys())}")
    print(f"     first EXAMPLE_ID = {rows[0]['EXAMPLE_ID']!r}")
EOF

echo "======================================================"
echo "Done  : $(date)"
echo "======================================================"
