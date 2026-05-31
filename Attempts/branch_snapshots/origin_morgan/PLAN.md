# Track 1 Implementation Plan
# Zero One Hack — Industrial AI (Infineon)

## Overview

Build and benchmark sequence models on semiconductor process data across three tasks:
1. **Next-step prediction** — given a partial sequence, predict the next step
2. **Sequence completion** — given a partial sequence (60% or 80% done), generate the full remainder
3. **Anomaly detection** — given a complete sequence, flag rule violations and attribute to a rule type

All three tasks are served by the **same trained model** (generative, autoregressive). A separate rule-checker handles rule attribution for Task 3. Results are logged to **Weights & Biases** during training and compared in a **Streamlit dashboard**.

---

## Path mapping for this repo

This plan was written against a generic `tracks/track_01/...` layout. This repo is laid out differently — translate paths as follows when running the commands below:

| Plan says | In this repo |
|---|---|
| `tracks/track_01/` | repo root |
| `tracks/track_01/training_data/` | `training_data/` |
| `tracks/track_01/solution/` | `solution/` (create at repo root for this experiment) |
| `tracks/track_01/training_data/generation_rules.md` | `training_data/generation_rules.md` ✅ exists |
| `tracks/track_01/training_data/generate_sequences.py` | `training_data/generate_sequences.py` ✅ exists |
| `tracks/track_01/eval_metrics.py` | **not in repo** — implement `solution/eval/metrics.py` from scratch (the spec is in `training_data/generation_rules.md` §5) |
| `ic_sequences.csv` / `igbt_sequences.csv` / `mosfet_sequences.csv` | `IC_variants.csv` / `IGBT_variants.csv` / `MOSFET_variants.csv` (1,000 seqs each; extra generated data in `{IC,IGBT}_generated_extra.csv`) |

**Other notes for this repo:**
- The CSVs use columns `SEQUENCE_ID, STEP` (long format) — as the plan assumes.
- `eval_input_valid.csv` / `eval_input_anomaly.csv` are organizer-distributed and **not in the repo yet**; point `run_eval.py --eval_dir` at them once you have them.
- Run with **pixi**, not a raw venv: prefix the plan's `python ...` commands with `pixi run`. The pixi env already includes torch (cu121), transformers, wandb, etc. (see [PIXI.md](PIXI.md)); `peft`, `streamlit`, `altair`, `editdistance`, `nltk` from the plan's `requirements.txt` are not yet in `pixi.toml` — add them before Steps 5/9.
- There is a separate, already-built Qwen fine-tune under `small_transformer_model/`. This plan is an **independent from-scratch experiment**; keep its outputs under `solution/` so the two don't collide.

---

## Step 0 — Before writing any code, read the repo

```
# Paths Claude Code should read first (relative to repo root):
tracks/track_01/README.md
tracks/track_01/training_data/generation_rules.md
tracks/track_01/training_data/generate_sequences.py
tracks/track_01/eval_metrics.py          # may already exist — check interface before reimplementing
```

Also inspect one CSV from each family to understand exact column names and step vocabulary:
```
tracks/track_01/training_data/ic_sequences.csv      # or similar filename
tracks/track_01/training_data/igbt_sequences.csv
tracks/track_01/training_data/mosfet_sequences.csv
```

The long-format training data has columns: `SEQUENCE_ID`, `STEP` (one row per step, ordered).
The eval files have: `eval_input_valid.csv` and `eval_input_anomaly.csv` — read and understand their schema before building the eval harness.

---

## Project Structure

Create everything under `tracks/track_01/solution/`:

```
solution/
├── data/
│   ├── __init__.py
│   ├── loader.py          # load CSVs → list of sequences
│   ├── vocab.py           # build step vocabulary, encode/decode
│   └── generator.py       # wrapper to call generate_sequences.py programmatically
├── models/
│   ├── __init__.py
│   ├── ngram.py           # n-gram baseline (n=2,3,4)
│   ├── transformer.py     # small GPT-style transformer trained from scratch
│   └── llm.py             # HuggingFace fine-tuned small LLM
├── tasks/
│   ├── __init__.py
│   ├── next_step.py       # Task 1: top-k next step prediction
│   ├── completion.py      # Task 2: full sequence completion
│   └── anomaly.py         # Task 3: perplexity scoring + rule attribution
├── eval/
│   ├── __init__.py
│   ├── metrics.py         # compute all metrics for all 3 tasks
│   └── rules.py           # programmatic rule checker (10 rule types from generation_rules.md)
├── train/
│   ├── train_transformer.py   # train small transformer from scratch
│   └── finetune_llm.py        # fine-tune HuggingFace LLM
├── app.py                 # Streamlit dashboard
├── run_baseline.py        # end-to-end: n-gram baseline on all 3 tasks, save results
├── run_eval.py            # run any trained model on eval set, save results
└── requirements.txt
```

---

## Step 1 — Data loading (`data/loader.py`)

```python
# Interface Claude Code should implement:

def load_sequences(csv_path: str) -> list[list[str]]:
    """Load long-format CSV → list of sequences. Each sequence is a list of step strings."""

def load_all_families(data_dir: str) -> dict[str, list[list[str]]]:
    """Returns {"IC": [...], "IGBT": [...], "MOSFET": [...]}"""

def train_val_split(sequences: list, val_ratio=0.1, seed=42) -> tuple[list, list]:
    """Deterministic split. Same seed everywhere."""
```

---

## Step 2 — Vocabulary (`data/vocab.py`)

The step vocabulary is small (~50–200 unique step names). Build a simple integer vocabulary from all training sequences.

```python
class Vocab:
    PAD = 0
    BOS = 1   # beginning of sequence
    EOS = 2   # end of sequence
    UNK = 3
    OFFSET = 4  # real steps start at index 4

    def __init__(self, sequences: list[list[str]]): ...
    def encode(self, steps: list[str]) -> list[int]: ...
    def decode(self, ids: list[int]) -> list[str]: ...
    def save(self, path: str): ...
    def load(cls, path: str) -> "Vocab": ...

# Always prepend BOS and append EOS when encoding a full sequence.
# Save vocab to solution/vocab.json so all models use the same token IDs.
```

---

## Step 3 — N-gram baseline (`models/ngram.py`)

Implement for n = 2, 3, 4. Default to n=3 (trigram) as primary baseline.

```python
class NGramModel:
    def __init__(self, n: int = 3): ...

    def fit(self, sequences: list[list[str]]): ...
    # Build count tables over n-grams of step strings (not integers — keep readable).

    def next_step_probs(self, context: list[str]) -> dict[str, float]:
        """Given last n-1 steps, return probability distribution over vocabulary.
        Back off to (n-1)-gram if context unseen, then unigram."""

    def top_k_next(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (step, prob) pairs."""

    def complete_sequence(self, partial: list[str], max_steps: int = 200) -> list[str]:
        """Greedy completion: always pick argmax until EOS or max_steps."""

    def sequence_log_prob(self, sequence: list[str]) -> float:
        """Sum of log P(step_i | context) over all steps. Used for anomaly scoring."""

    def save(self, path: str): ...
    def load(cls, path: str) -> "NGramModel": ...
```

**Tasks 1, 2, 3 from n-gram:**
- Task 1: `top_k_next(context[-n+1:], k=5)` — report top-1/3/5 accuracy and MRR
- Task 2: `complete_sequence(partial)` — compare to ground truth with edit distance
- Task 3: `sequence_log_prob(sequence)` — lower = more anomalous; pick threshold on validation set

---

## Step 4 — Small transformer from scratch (`models/transformer.py`)

This is the primary neural model. Keep it small enough to train in a few hours on A100s.

**Architecture:**
```
Vocab size:   len(vocab) + 4 special tokens  (~200)
d_model:      256
n_heads:      4
n_layers:     6
d_ff:         1024
max_seq_len:  256   (sequences are 115–150 steps + BOS/EOS)
dropout:      0.1
```

```python
import torch
import torch.nn as nn

class ProcessTransformer(nn.Module):
    """Causal (decoder-only) transformer for process sequence modeling."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, max_seq_len=256, dropout=0.1): ...

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns logits of shape (batch, seq_len, vocab_size)."""

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation with top-k sampling."""

    def sequence_log_prob(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-sequence sum of log P(token_i | token_<i). Shape: (batch,)"""
```

**Training (`train/train_transformer.py`):**
- Loss: cross-entropy on next-token prediction (shift input by 1)
- Optimizer: AdamW, lr=1e-3 with cosine decay to 1e-4
- Batch size: 64 sequences (pad to max length in batch)
- Epochs: train until validation loss plateaus (watch for ~20–50 epochs on 3000 sequences)
- Checkpoint: save best model by validation loss every 5 epochs
- Log to WandB: training loss, validation loss, top-1 accuracy on validation set

```bash
python train/train_transformer.py \
  --data_dir ../training_data \
  --output_dir ./checkpoints/transformer \
  --d_model 256 --n_layers 6 --n_heads 4 \
  --epochs 50 --batch_size 64 --lr 1e-3 \
  --wandb_project zero-one-hack --wandb_run transformer_256_6l
```

**Also train a larger variant for scaling comparison:**
```bash
python train/train_transformer.py \
  --d_model 512 --n_layers 8 --n_heads 8 \
  --wandb_run transformer_512_8l
```

---

## Step 5 — Fine-tuned LLM (`models/llm.py`)

Use a small pretrained HuggingFace model. Recommended: `Qwen/Qwen2.5-0.5B` or `HuggingFaceTB/SmolLM2-360M`.

**Sequence formatting for LLM:**
```
Represent each sequence as plain text, steps separated by " -> ":
"RECEIVE WAFER LOT -> LOT IDENTIFICATION -> INITIAL WAFER INSPECTION -> MEASURE THICKNESS -> ..."
```

**Fine-tuning script (`train/finetune_llm.py`):**
- Use HuggingFace `Trainer` with causal LM objective
- The LLM tokenizer already handles strings — no custom vocab needed
- Use PEFT/LoRA if GPU memory is tight (rank=16, alpha=32, target modules: q_proj, v_proj)
- Batch size: 4–8, gradient accumulation: 8 steps, effective batch = 32–64
- lr: 2e-4 with warmup 50 steps, cosine decay
- Max sequence length: 1024 tokens (a full sequence in text is roughly 500–800 tokens)
- Log to WandB: same project, run name `llm_qwen_lora`

```python
# Interface for inference:
class LLMModel:
    def __init__(self, model_name: str, checkpoint_path: str = None): ...

    def top_k_next(self, partial_sequence: list[str], k: int = 5) -> list[tuple[str, float]]:
        """Format partial as text, get logits at last token, return top-k step names."""

    def complete_sequence(self, partial: list[str], max_new_steps: int = 200) -> list[str]:
        """Generate continuation, parse " -> " delimited steps from output."""

    def sequence_log_prob(self, sequence: list[str]) -> float:
        """Sum of log P(token | context) for all tokens. Used for anomaly scoring."""
```

---

## Step 6 — Rule checker for anomaly attribution (`eval/rules.py`)

This is what separates a good Task 3 score from a great one. After the model flags a sequence as anomalous, the rule checker identifies *which rule* was violated.

```python
# Read generation_rules.md carefully before implementing.
# Implement each of the 10 rule types as a Python function.

RULE_TYPES = [
    "missing_prerequisite",
    "wrong_order",
    "invalid_step_for_family",
    "duplicate_step",
    "missing_mandatory_step",
    "step_in_wrong_phase",
    "invalid_transition",
    "parameter_out_of_range",  # if parameters are in the data
    "incomplete_block",
    "forbidden_sequence",
]

def check_all_rules(sequence: list[str], family: str, vocab: Vocab) -> dict[str, bool]:
    """Run all 10 rule checks. Returns {rule_name: violated}."""

def attribute_anomaly(sequence: list[str], family: str, vocab: Vocab) -> str | None:
    """Return the name of the first violated rule, or None if sequence is valid."""
```

**Important:** base the rule implementations directly on `generation_rules.md`, not on intuition. Read it carefully — the grammar is the ground truth.

---

## Step 7 — Metrics (`eval/metrics.py`)

Implement all metrics for all 3 tasks. Match the interface of `eval_metrics.py` if it already exists in the repo, or implement from scratch.

```python
# Task 1
def next_step_metrics(predictions: list[list[tuple[str, float]]],
                      ground_truth: list[str]) -> dict:
    """predictions: list of top-k (step, prob) lists. Returns top-1/3/5 accuracy, MRR."""

# Task 2
def completion_metrics(predictions: list[list[str]],
                       ground_truth: list[list[str]]) -> dict:
    """Returns exact_match_rate, mean_normalized_edit_distance,
       token_accuracy, block_level_accuracy."""

# Task 3
def anomaly_metrics(scores: list[float], labels: list[int],
                    threshold: float,
                    predicted_rules: list[str | None],
                    true_rules: list[str | None]) -> dict:
    """labels: 0=valid, 1=anomaly.
       Returns binary_accuracy, precision, recall, f1, roc_auc,
       confusion_matrix, rule_attribution_accuracy."""

def find_threshold(scores: list[float], labels: list[int]) -> float:
    """Find threshold maximising F1 on the eval set."""
```

---

## Step 8 — End-to-end eval runner (`run_eval.py`)

```bash
# Run any model against the eval set:
python run_eval.py \
  --model ngram \
  --n 3 \
  --eval_dir ../  # looks for eval_input_valid.csv and eval_input_anomaly.csv here
  --output results/ngram_3.json

python run_eval.py \
  --model transformer \
  --checkpoint checkpoints/transformer/best.pt \
  --output results/transformer_256_6l.json

python run_eval.py \
  --model llm \
  --checkpoint checkpoints/llm_qwen \
  --output results/llm_qwen.json
```

All results saved as JSON with structure:
```json
{
  "model": "transformer_256_6l",
  "task1": {"top1": 0.82, "top3": 0.94, "top5": 0.97, "mrr": 0.87},
  "task2": {"exact_match": 0.41, "edit_distance": 0.12, "token_acc": 0.79, "block_acc": 0.65},
  "task3": {"f1": 0.88, "roc_auc": 0.93, "rule_attribution_acc": 0.71},
  "by_family": {...},
  "timestamp": "..."
}
```

---

## Step 9 — Streamlit dashboard (`app.py`)

Three tabs:

**Tab 1: Model comparison table**
- Load all JSON files from `results/`
- Show side-by-side table of all metrics across all models
- Highlight best value per metric in green

**Tab 2: Live demo**
- Family selector (IC / IGBT / MOSFET)
- Text area: paste or type a partial sequence (one step per line, or " -> " separated)
- Model selector (ngram / transformer / llm)
- Button: "Predict next step" → show top-5 with probabilities as a bar chart
- Button: "Complete sequence" → show full predicted continuation
- Button: "Check for anomalies" → show anomaly score + rule attribution if flagged

**Tab 3: Training curves**
- If WandB is configured, pull run history via `wandb.Api()` and plot loss curves
- Fallback: load training logs from `checkpoints/*/training_log.jsonl` and plot with Altair

```bash
streamlit run solution/app.py
```

---

## Step 10 — WandB setup

```python
# At the top of each training script:
import wandb

wandb.init(
    project="zero-one-hack",
    name=run_name,
    config={
        "model": model_type,
        "d_model": d_model,
        "n_layers": n_layers,
        "lr": lr,
        "batch_size": batch_size,
        "data": "IC+IGBT+MOSFET 3000 sequences",
    }
)

# During training loop:
wandb.log({"train/loss": loss, "val/loss": val_loss, "val/top1_acc": top1, "epoch": epoch})

# After eval:
wandb.log({
    "eval/task1_top1": ...,
    "eval/task2_exact_match": ...,
    "eval/task3_f1": ...,
    "eval/task3_roc_auc": ...,
})

wandb.finish()
```

If WandB is not available (no internet / no API key), fall back to writing logs to `checkpoints/{run_name}/training_log.jsonl`, one JSON object per epoch.

---

## Requirements (`requirements.txt`)

```
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.18.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
editdistance>=0.6.0
wandb>=0.17.0
streamlit>=1.35.0
altair>=5.0.0
nltk>=3.8.0
tqdm>=4.66.0
```

---

## Execution order

```bash
cd tracks/track_01/solution

pip install -r requirements.txt

# 1. Run baseline immediately (30–60 min)
python run_baseline.py
# → results/ngram_2.json, ngram_3.json, ngram_4.json

# 2. Train small transformer (2–4 hours on A100)
python train/train_transformer.py --d_model 256 --n_layers 6 --wandb_run transformer_small
python train/train_transformer.py --d_model 512 --n_layers 8 --wandb_run transformer_large

# 3. Fine-tune LLM (2–4 hours on A100)
python train/finetune_llm.py --model_name Qwen/Qwen2.5-0.5B --wandb_run llm_qwen_0.5b

# 4. Evaluate all models
python run_eval.py --model ngram --n 3 --output results/ngram_3.json
python run_eval.py --model transformer --checkpoint checkpoints/transformer_small/best.pt --output results/transformer_small.json
python run_eval.py --model transformer --checkpoint checkpoints/transformer_large/best.pt --output results/transformer_large.json
python run_eval.py --model llm --checkpoint checkpoints/llm_qwen --output results/llm_qwen.json

# 5. Launch dashboard
streamlit run app.py
```

---

## Key implementation notes for Claude Code

- **Tokenize at the step level**, not character/subword level, for the custom transformer. Each unique step name is one token. The LLM uses its own tokenizer on the text-formatted sequences.
- **Padding**: pad sequences to the longest in each batch. Use `attention_mask` to exclude padding from loss.
- **Evaluation during training**: run Task 1 metrics (top-1 accuracy) on the held-out 10% validation split every 5 epochs. This is fast and gives early signal.
- **Anomaly threshold**: do NOT use the eval set to pick the threshold — use the validation split. Then apply the threshold to the eval set.
- **Rule attribution**: implement rule checks based exactly on `generation_rules.md`. Read it fully before writing any rule checker code.
- **Reproducibility**: set `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` at the top of every training script.
- **Checkpointing on cluster**: save checkpoints every 5 epochs explicitly. Leonardo jobs can be preempted.
- **Family tags**: sequences from different families should be tagged. Test whether adding a family token at the start of each sequence (e.g. `<IC>`, `<IGBT>`, `<MOSFET>`) improves performance — it probably will for the transformer.
```
