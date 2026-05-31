#!/bin/bash
#SBATCH --job-name=gpt-guided
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/gpt_v2_%j.log
#SBATCH --error=outputs/slurm/gpt_v2_%j.log

echo "======================================================"
echo "Job     : $SLURM_JOB_ID  (gpt-guided v2)"
echo "Node    : $SLURMD_NODENAME"
echo "Start   : $(date)"
echo "======================================================"

module load python/3.11.7
module load cuda/12.1
source ~/venv/bin/activate

cd ~/zero_one_hack_0625
mkdir -p outputs/slurm results

export TOKENIZERS_PARALLELISM=false

# ── Quick sanity check: how many sequences do we have now? ──────────────────
echo ""
echo "=== Dataset sizes ==="
python3 - << 'EOF'
from data import load_family, FAMILIES
for fam in FAMILIES:
    seqs = load_family(fam)
    print(f"  {fam}: {len(seqs)} sequences")
EOF

# ── STEP 1: Grammar-guided eval of existing GPT:small checkpoints ───────────
# (no training needed — tests the guide immediately on existing models)
echo ""
echo "=== [1/5] Grammar guide eval — GPT:small (existing checkpoints) ==="
python eval_guided.py --kind gpt --size small --eval-seqs 300

# ── STEP 2: Train GPT:large — all families (ID run) ─────────────────────────
echo ""
echo "=== [2/5] Train GPT:large — all families ==="
python train.py \
    --model gpt:large \
    --train-families mosfet igbt ic \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_large_allfam

# ── STEP 3: Train GPT:large — 3 × leave-one-out ─────────────────────────────
echo ""
echo "=== [3/5] Train GPT:large — LOO MOSFET ==="
python train.py \
    --model gpt:large \
    --leave-out mosfet \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_large_loo_mosfet

echo ""
echo "=== [4/5] Train GPT:large — LOO IGBT ==="
python train.py \
    --model gpt:large \
    --leave-out igbt \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_large_loo_igbt

echo ""
echo "=== Train GPT:large — LOO IC ==="
python train.py \
    --model gpt:large \
    --leave-out ic \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --out outputs/gpt_large_loo_ic

# ── STEP 4: Grammar-guided eval of GPT:large ────────────────────────────────
echo ""
echo "=== [5/5] Grammar guide eval — GPT:large (new checkpoints) ==="
python eval_guided.py --kind gpt --size large --eval-seqs 300

# ── STEP 5: Run full submission pipeline ────────────────────────────────────
echo ""
echo "=== Full eval pipeline (submission CSVs) ==="
python eval_runner.py \
    --model gpt:large \
    --checkpoint outputs/gpt_large_allfam \
    --score

echo "======================================================"
echo "Done: $(date)"
echo "======================================================"
