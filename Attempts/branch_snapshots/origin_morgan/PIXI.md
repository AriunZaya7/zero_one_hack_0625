# Running on Leonardo with pixi

This repo uses [pixi](https://pixi.sh/) for a single, reproducible environment.
The environment is defined in [`pixi.toml`](pixi.toml): Python 3.11 plus the full ML
stack (torch, transformers, accelerate, wandb, datasets, …), with the PyTorch
**CUDA 12.1** index pinned so you get GPU wheels on Leonardo's A100s.

> `platforms = ["linux-64"]` — that's the cluster. `pixi install` / `pixi run`
> run on the Linux login/compute nodes, even though you may edit on Windows.

## Tasks

| Task | What it does |
| --- | --- |
| `pixi run gpu-check` | Prints torch version + whether CUDA is visible |
| `pixi run debug` | Tiny CPU smoke test of the training script |
| `pixi run train` | Full Qwen2-0.5B fine-tune → `small_transformer_model/checkpoints/best_model` |
| `pixi run eval` | Tests the trained model (next-step + completion, per-family, ID→OOD) → `small_transformer_model/eval/eval_results.json` |
| `pixi run eval-smoke` | Fast sanity check of the eval harness (8 examples) |
| `pixi run ngram-baseline` | N-gram next-step baseline, for the baseline-vs-trained comparison |

You can also run any file directly: `pixi run python path/to/script.py`.

## Workflow

### 1. On a login node (has internet)

Install the environment and pre-cache the Qwen weights, so the (air-gapped)
compute nodes never need to download anything:

```bash
cd ~/zero_one_hack_0625            # wherever you cloned/scp'd the repo
pixi install                       # solves + downloads the env into ./.pixi

# pre-download Qwen2-0.5B into the HuggingFace cache (~1 GB)
pixi run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True); \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)"
```

### 2. Smoke-test cheaply (login node / CPU is fine)

```bash
pixi run gpu-check                 # 'cuda available: False' on a login node is expected
pixi run debug                     # tiny training run, catches bugs fast
pixi run eval-smoke                # 8 sequences through the test harness
```

> `gpu-check` only reports `True` inside a Slurm GPU job — login nodes have no GPU.

### 3. Train + test inside a Slurm GPU job

Use the hackathon reservation (`boost_usr_prod`, `s_tra_ncc`). Minimal job script:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

cd $SLURM_SUBMIT_DIR
export RUN_COMMAND="$(which pixi) run"

$RUN_COMMAND train
$RUN_COMMAND eval
```

Submit and monitor:

```bash
sbatch job.slurm
squeue --me
tail -f slurm-<job_id>.out
```

## Notes

- **`pixi install` must run on a login node** — compute nodes are air-gapped and
  the solve/download will fail there.
- The trained model is loaded from a **local checkpoint** during `eval`, so the
  eval step needs no internet and runs fine on a compute node.
- **W&B on air-gapped nodes:** set `export WANDB_MODE=offline` in the job, then
  `wandb sync` from the login node afterwards.
