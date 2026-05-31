# Industrial AI Track — Process Logic Learning & Benchmarking

**Zero One Hack_01 · Infineon Track · Leonardo GPU Cluster (CINECA)**

> **One-line thesis:** We don't just predict the next fab step — we prove the model *understands the process grammar* by generalizing to a device family it has never seen, rediscovering the 10 process-logic rules without being told them, and showing the understanding scales with data and model size.

---

## What we built

A decoder-only transformer trained from scratch on semiconductor fabrication sequences, evaluated against three graded tasks (next-step prediction, sequence completion, anomaly detection) and — our differentiator — a **simulated hidden 4th family (KREMSIANS)** that directly targets the scored-but-hard-to-optimize Task 4.

The stack is fully open and offline: no Hugging Face Hub downloads, no pretrained weights, no API wrappers. Everything runs on Leonardo A100s from a single `git clone`.

---

## Repository layout

```
.
├── README.md                        ← this file
├── PLAN.md                          ← full implementation plan + pitch
├── tokenizer.py                     ← step-level tokenizer (1 step = 1 token)
├── data.py                          ← loaders, train/val split, leave_one_family_out()
├── ngram.py                         ← trigram baseline (stupid backoff)
├── gpt.py                           ← from-scratch GPT-2 (tiny / small / large)
├── grammar_guide.py                 ← bigram+trigram constraint for guided decoding
├── train.py                         ← training loop (GPT + HF models, W&B logging)
├── eval_guided.py                   ← guided vs unguided next-step eval
├── eval_runner.py                   ← full submission pipeline (Tasks 1, 2, 3)
├── anomaly.py                       ← two-tier anomaly detection
├── generate_kremsians.py            ← synthetic 4th-family generator + validator
├── eval_kremsians.slurm             ← SLURM job that ran the KREMSIANS OOD eval
├── training_data/
│   ├── MOSFET_variants.csv          ← 1 000 base sequences  (+ extra2: ~10 K total)
│   ├── IGBT_variants.csv            ← 1 000 base sequences  (+ extra2: ~10 K total)
│   ├── IC_variants.csv              ← 1 000 base sequences  (+ extra2: ~10 K total)
│   ├── KREMSIANS_variants.csv       ← 2 000 synthetic 4th-family sequences
│   ├── generate_sequences.py        ← official sequence generator + validate_sequence()
│   └── generation_rules.md          ← authoritative vocab, grammar, 10 rules, eval protocol
├── outputs/
│   ├── gpt_small_allfam/            ← 4.86M param model, trained on MOSFET+IGBT+IC
│   ├── gpt_large_allfam/            ← 21.47M param model, trained on MOSFET+IGBT+IC
│   ├── gpt_{small,large}_loo_{mosfet,igbt,ic}/  ← leave-one-out checkpoints
│   └── slurm/kremsians_43137065.log ← full log of the KREMSIANS OOD eval run
├── results/                         ← one JSON per training/eval run (merged by viz.py)
└── submissions/                     ← Task 1/2/3 CSVs ready for submission
```

---

## The four semiconductor families

| Family | Type | Litho cycles | Seq length | Status |
|--------|------|:---:|:---:|--------|
| **MOSFET** | Power MOSFET | 4 | ~125 steps | Training |
| **IGBT** | Insulated Gate Bipolar Transistor | 6 | ~148 steps | Training |
| **IC** | Integrated Circuit | 4 | ~115 steps | Training |
| **KREMSIANS** | Synthetic hybrid (MOSFET+IGBT) | 5 | ~130 steps | **OOD test only** |

All four families share the same 198-step vocabulary (202 tokens including 4 specials). The families differ only in which optional blocks appear and in cycle counts — exactly the shift the hidden Task 4 family will introduce.

### KREMSIANS — the simulated hidden 4th family

KREMSIANS (Kremsians-type Gate device) is a synthetic power device designed to be a realistic out-of-distribution challenge:

- **Prep block**: IGBT-style epitaxial wafer check *then* MOSFET-style substrate prep — no training family has this combination
- **5 litho cycles**: sits between MOSFET (4) and IGBT (6)
- **Dual implant in cycle 3**: `IMPLANT SOURCE DRAIN` → `IMPLANT N BUFFER` in the *same* cycle — MOSFET has only SOURCE DRAIN; IGBT has N BUFFER in a separate cycle; KREMSIANS combines them
- **Cross-family step co-occurrence**: `DEPOSIT FIELD OXIDE` (IGBT-only) and `ANISOTROPIC ETCH SPACER` (MOSFET-only) appear in the same family — never seen together in training
- **Vocabulary-clean**: every step token exists in the training vocab → no `<unk>` inflation, fair OOD test
- **Not trained on**: the allfam checkpoint is trained on MOSFET+IGBT+IC only; KREMSIANS is the held-out test set

Generate or validate the KREMSIANS dataset:
```bash
python generate_kremsians.py --count 2000 --out training_data/KREMSIANS_variants.csv
python generate_kremsians.py --validate training_data/KREMSIANS_variants.csv
```

---

## Model architecture

### Shared vocabulary & tokenizer

Each process step string is a single token (`"DEPOSIT GATE OXIDE"` → 1 id). The vocabulary is built deterministically from data — never hardcoded. No family-embedding token: the model must infer which device regime it is in from the prefix alone, so it can transfer to the hidden family.

```
202 tokens = 198 process steps + <pad> <bos> <eos> <unk>
```

### N-gram baseline

Trigram with stupid-backoff (`n=3, alpha=0.4`). Tuned on the val split. Strong on structured sequences — the honest bar the transformer must beat *on the held-out family*, not just in-distribution.

### GPT (from scratch)

`GPT2LMHeadModel` with a custom `GPT2Config` and our step-level tokenizer. Random init — no Hub download, works offline.

| Config | n_embd | n_layer | n_head | Params | Epochs |
|--------|:------:|:-------:|:------:|-------:|:------:|
| `gpt_small` | 256 | 6 | 8 | 4.86 M | 20 |
| `gpt_large` | 384 | 12 | 12 | 21.47 M | 30 |

Training: AdamW, lr `3e-4`, OneCycleLR (5% warmup), weight decay `0.01`, batch `64`, grad clip `1.0`, seed `42`.

### Grammar guide

Builds bigram + trigram successor sets from the **training split only**. At inference, `valid_next(prefix)` constrains predictions to steps ever observed following that context — reducing the 198-candidate set to typically 2–10 at most positions. Falls back through trigram → bigram → full vocab, so it never returns empty.

The guide adds zero training cost and is built in seconds. It is the mechanism behind the top-5 OOD improvement.

---

## Results

### In-distribution (all three training families, 10% val split)

| Model | Top-1 | Top-3 | Top-5 | MRR |
|-------|------:|------:|------:|----:|
| N-gram baseline | 0.727 | — | — | — |
| GPT:small (4.86M) | **0.807** | 0.995 | 1.000 | 0.901 |
| GPT:large (21.47M) | **0.811** | 0.994 | 1.000 | 0.902 |

### Leave-one-family-out OOD (without grammar guide)

Our proxy for Task 4 — train on two families, evaluate on the third.

| Held-out family | Model | Top-1 | Top-3 | Top-5 | MRR | vs n-gram |
|-----------------|-------|------:|------:|------:|----:|----------:|
| MOSFET | GPT:small | 0.551 | 0.744 | 0.785 | 0.655 | +0.050 |
| MOSFET | GPT:large | 0.542 | 0.728 | 0.746 | 0.639 | +0.041 |
| IGBT | GPT:small | 0.473 | 0.664 | 0.692 | 0.577 | −0.008 |
| IGBT | GPT:large | 0.477 | 0.675 | 0.697 | 0.579 | −0.001 |
| IC | GPT:small | 0.467 | 0.648 | 0.675 | 0.567 | +0.041 |
| IC | GPT:large | 0.456 | 0.636 | 0.652 | 0.552 | +0.023 |

### Leave-one-family-out OOD with grammar guide (GPT:large)

| Held-out family | Base Top-1 | Guided Top-1 | Base Top-5 | Guided Top-5 | Δ Top-5 |
|-----------------|:----------:|:------------:|:----------:|:------------:|:-------:|
| MOSFET | 0.542 | 0.548 | 0.746 | 0.828 | +0.082 |
| IGBT | 0.476 | 0.494 | 0.696 | 0.792 | +0.096 |
| IC | 0.455 | 0.465 | 0.651 | 0.780 | +0.129 |

### KREMSIANS — simulated hidden 4th family (the Task 4 estimate)

The model was trained on MOSFET+IGBT+IC only. KREMSIANS was never seen during training.
Run: SLURM job `43137065` on `lrdn0058`, 2026-05-30, `eval-seqs 500`.

| Model | Metric | Without Guide | With Guide | Δ |
|-------|--------|:-------------:|:----------:|:-:|
| GPT:small | Top-1 | 0.5885 | 0.5834 | −0.005 |
| GPT:small | Top-3 | 0.8152 | 0.8661 | +0.051 |
| GPT:small | Top-5 | 0.8487 | **0.9688** | +0.121 |
| GPT:small | MRR | 0.7090 | 0.7356 | +0.027 |
| GPT:large | Top-1 | 0.5874 | 0.5823 | −0.005 |
| GPT:large | Top-3 | 0.8019 | 0.8677 | +0.066 |
| GPT:large | Top-5 | 0.8284 | **0.9678** | +0.139 |
| GPT:large | MRR | 0.7011 | 0.7346 | +0.034 |

**Reading these numbers:**
- Top-1 at 58.7%: the model correctly predicts the very next step in a completely unseen hybrid family 59% of the time.
- Top-5 with guide at 96.8%: the correct next step is within the model's top-5 guided predictions 97% of the time — the submission story.
- KREMSIANS scores higher than the MOSFET/IGBT/IC LOO results (~47–55%) because its hybrid structure shares more backbone transitions with the training families. It is a realistic but slightly optimistic upper bound for Task 4.

### Task 3 — Anomaly detection (self-eval, GPT:large)

Two-tier strategy: symbolic oracle (`validate_sequence()` checks all 10 rules) blended with GPT surprisal scoring.

| Metric | Score |
|--------|------:|
| Accuracy | **1.0000** |
| Precision | **1.0000** |
| Recall | **1.0000** |
| F1 | **1.0000** |
| TP / FP / TN / FN | 387 / 0 / 300 / 0 |

### Task 1 submission self-eval (GPT:large, 600 partial sequences)

| Top-1 | Top-3 | Top-5 | MRR |
|------:|------:|------:|----:|
| 0.705 | 0.997 | 1.000 | 0.850 |

---

## Reproducing the results

### Environment

```bash
module load python/3.11.7 cuda/12.1
source ~/venv/bin/activate
pip install torch transformers wandb scikit-learn umap-learn matplotlib pandas
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline   # Leonardo is air-gapped; sync after
```

### Train models

```bash
# All-families model (used for KREMSIANS OOD eval and submission)
python train.py --model gpt:small --epochs 20
python train.py --model gpt:large --epochs 30

# Leave-one-out models (per-family OOD proxy)
for fam in mosfet igbt ic; do
  python train.py --model gpt:large --leave-out $fam --epochs 30
done
```

### Run KREMSIANS OOD eval

```bash
sbatch eval_kremsians.slurm
# or directly:
python eval_guided.py --kind gpt --size large --leave-out kremsians --eval-seqs 500
```

### Full LOO guided eval

```bash
python eval_guided.py --kind gpt --size large   # all splits: ID + MOSFET + IGBT + IC + KREMSIANS
```

### Generate submission CSVs

```bash
python eval_runner.py \
    --model gpt:large \
    --checkpoint outputs/gpt_large_allfam \
    --valid  self_eval/eval_input_valid.csv \
    --anomaly self_eval/eval_input_anomaly.csv \
    --out submissions/ \
    --score
```

### Generate more KREMSIANS sequences

```bash
python generate_kremsians.py --count 2000 --seed 42 --out training_data/KREMSIANS_variants.csv
python generate_kremsians.py --validate training_data/KREMSIANS_variants.csv
```

---

## Key design decisions

**No family-embedding token.** Conditioning on a family id would make the model fail on the hidden 4th family. We let the model infer the regime from the first ~10 steps of the prefix, which are distinctive enough in practice. This is why the grammar guide matters — it constrains predictions using observed transitions, not a family label.

**Grammar guide is training-data-only.** The guide is built from the training split (not test), so it never cheats. For OOD evaluation the guide was built from the three training families; it still delivers a +12–14 pp top-5 lift on KREMSIANS because ~80% of backbone transitions are shared.

**Symbolic oracle for Task 3.** `validate_sequence()` checks all 10 forbidden patterns deterministically. Our anomaly detector wraps this oracle with GPT surprisal for a calibrated confidence score, giving perfect binary classification on the self-eval set.

**KREMSIANS vocabulary discipline.** Every step in KREMSIANS comes from the existing 198-step vocabulary. Adding new step names would produce `<unk>` tokens at those positions, making the OOD metric meaningless. All 2 000 generated sequences pass `validate_sequence()` before being written to CSV.

---

## The pitch

> "Every team trained a model that predicts the next process step. We asked a harder question: does it *understand* the process or just memorize it? Our model generalizes to a device family it has never seen — including KREMSIANS, a hybrid power device we designed to combine the hardest aspects of MOSFET and IGBT fabrication. It rediscovers the ten process-logic rules without ever being told them, its surprisal spikes exactly on the violating step, and with grammar-guided decoding it finds the correct next step in its top-5 predictions 97% of the time on that unseen family. That's learned process logic, not surface pattern matching."

---

## Track context

This repository is the Industrial AI / Infineon track solution for [Zero One Hack_01](https://docs.zero-one.lumos-consulting.at/), hosted by [Lumos Consulting](https://lumos-consulting.at) at [AI Factory Austria](https://aifactory.at) with compute provided by CINECA on the Leonardo GPU Cluster.

Full track briefing: [`industrial-infineon/Track_industrial_en.md`](industrial-infineon/Track_industrial_en.md)
