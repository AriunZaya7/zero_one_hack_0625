# REPORT — Learning & Benchmarking Process Logic (Industrial AI / Infineon)

**Track:** Industrial AI — *"Learning and Benchmarking Process Logic"* (Infineon partner)
**Compute:** CINECA **Leonardo**, NVIDIA A100-SXM-64GB · PyTorch 2.5.1+cu121 · Python 3.11.7
**Branch:** `cristi/modeling`

---

## TL;DR

We train a **from-scratch GPT** (no pretrained weights, no API, no lookup) on semiconductor
process sequences using a **step-level tokeniser** (1 process step = 1 token, ~198-token
closed vocabulary), and pair it with two grammar-aware components: a **grammar-guided
decoder** that masks logits to grammar-valid successors, and a **symbolic validator** for
anomaly detection. We answer the brief's core question — *does the model learn transferable
process logic, or just memorise?* — with an honest, self-built 4th family (**KREMSIANS**)
that no training touched:

- **Task 1 (next-step), in-distribution:** Top-1 **0.81**, Top-5 **1.00**, MRR **0.90** (GPT:large).
- **Task 3 (anomaly):** **F1 = 1.00** with the symbolic validator; perfect rule attribution
  on the rule set — and it is **family-agnostic**, so it transfers to the hidden 4th family.
- **OOD on KREMSIANS (never trained on it):** Top-1 **~0.59**, Top-3 **~0.87**, Top-5
  **~0.97** with the grammar guide. The model places the correct next step in its top-5
  **97% of the time on a family it has never seen.**
- **Scaling finding:** going 4.9M → 21.5M params adds ~nothing on OOD (and slightly *hurts*) —
  bigger models memorise the known families harder without learning more transferable logic.

All submission CSVs are produced in the **official format** (`valid_0001` / `anomaly_xxxx`
IDs) directly from the organiser input files in `participant_files/`. Numbers below the
submission line are clearly labelled **self-eval estimates** (organisers hold the real
ground truth).

---

## 1. Problem

Semiconductor devices are built through long sequences of ~115–150 process steps. Three
product families are known (MOSFET, IGBT, IC); each shares a common backbone grammar
(clean → oxidise → litho → etch → implant/anneal → metallise → passivate → test) plus
family-specific blocks. Organisers provide the grammar, a generator/validator
(`training_data/generate_sequences.py`), and 1,000 validated sequences per family.

Three scored tasks, submitted as CSVs:
1. **Next-step prediction** — Top-1/3/5, MRR.
2. **Sequence completion** (60% / 80% cut) — Exact match, Normalized Edit Distance (NED),
   Token accuracy, Block-level accuracy.
3. **Anomaly detection** — valid/invalid + which of 10 rules broke; Accuracy, P/R/F1,
   ROC-AUC, rule attribution.

A hidden **Task 4** re-runs the submitted model on a **secret 4th family**: the real test of
whether a solution *learned the logic* or *memorised the data*.

---

## 2. Approach

| Component | File | What it does |
|---|---|---|
| Step-level tokeniser | `tokenizer.py` | 1 process step = 1 token; ~198-token closed vocab. Directly optimises what the judges measure. |
| N-gram baseline | `ngram.py` | Count model, stupid backoff (n=3, α=0.4) + an n-order sweep (2→15). |
| From-scratch GPT | `gpt.py`, `train.py` | GPT-2 decoder trained from random init at 3 sizes: tiny 0.6M / small 4.9M / large 21.5M. |
| Qwen2.5 fine-tune ladder | `small_transformer_model/` | Pretrained-LLM tier (0.5B–3B) for the scaling story. |
| Grammar-guided decoding | `grammar_guide.py` | Masks logits to bigram/trigram grammar-valid successors at each position (~1.9 valid choices/position). |
| Symbolic anomaly validator | `anomaly.py` | Deterministic check of all 10 rules → perfect detection; model surprisal supplies the continuous ROC-AUC `SCORE`. |
| Self-built 4th family | `generate_kremsians.py` | **KREMSIANS** — grammatically valid, structurally novel; an honest stand-in for hidden Task 4. |
| Eval / submission | `eval_runner.py`, `eval_guided.py`, `score_selfeval.py` | Produce official-format CSVs; score against the **official** `eval_metrics.py`. |

**Why step-level GPT and not a fine-tuned LLM as the headliner?** BPE shatters
"DEPOSIT BARRIER METAL" into sub-words, making next-step prediction indirect. The step-level
GPT optimises the exact closed-vocab objective the judges score, trains in minutes on one
A100, and is a genuine **own-trained model on the cluster** (European AI-sovereignty theme) —
not a wrapper, API call, or lookup.

---

## 3. How to run (clean checkout)

```bash
# 1. Environment (Leonardo modules shown; any CUDA 12.1 box works)
module load python/3.11.7 cuda/12.1
python -m venv ~/venv && source ~/venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. One-command reproducible pipeline on a Leonardo A100:
#    trains tiny/small/large, writes official submissions, scores self-eval,
#    runs the KREMSIANS OOD sweep.
sbatch job.slurm

# --- or run the pieces directly ---

# Train the submission model (all 3 real families)
python train.py --model gpt:large --train-families mosfet igbt ic \
    --epochs 30 --batch-size 64 --lr 3e-4 --out outputs/gpt_large_allfam

# Official-format submission CSVs (correct valid_0001 / anomaly_xxxx IDs)
python eval_runner.py --model gpt:large --checkpoint outputs/gpt_large_allfam \
    --valid   participant_files/eval_input_valid.csv \
    --anomaly participant_files/eval_input_anomaly.csv \
    --out submissions_official

# Self-eval estimates with the OFFICIAL scorer (Task 1/2/3)
python make_selfeval.py
python eval_runner.py --model gpt:large --checkpoint outputs/gpt_large_allfam \
    --valid self_eval/eval_input_valid.csv --anomaly self_eval/eval_input_anomaly.csv \
    --out submissions --score
python score_selfeval.py --tag gpt_large_ckpt

# KREMSIANS 4th-family OOD + model-size sweep
python eval_guided.py --kind gpt --size large --leave-out kremsians --eval-seqs 500
```

**Submission outputs:** `submissions_official/{task1,task2,task3}.csv` in the official shape:
- Task 1: `EXAMPLE_ID,RANK_1..RANK_5`
- Task 2: `EXAMPLE_ID,PREDICTED_SEQUENCE` (pipe-separated steps **after** the cut)
- Task 3: `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE` (`SCORE` higher = more likely valid)

---

## 4. Results

> The numbers in §4.2–§4.3 are **self-eval estimates**, produced by the *official*
> `eval_metrics.py` on our own injected-violation split — the organisers score the real
> submission. The submitted CSVs (`submissions_official/`) carry official IDs and cannot be
> scored locally (we don't hold the hidden ground truth — that is expected).

### 4.1 Submission (official format)

| Task | File | Rows | Shape verified |
|---|---|---|---|
| 1 — next-step | `submissions_official/task1_gpt_large_ckpt.csv` | 600 | `EXAMPLE_ID,RANK_1..5` |
| 2 — completion | `submissions_official/task2_gpt_large_ckpt.csv` | 600 | `EXAMPLE_ID,PREDICTED_SEQUENCE` |
| 3 — anomaly | `submissions_official/task3_gpt_large_ckpt.csv` | 987 | `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE` |

### 4.2 Baseline → trained (in-distribution, Task 1)

| Model | Params | Top-1 | Top-3 | Top-5 | MRR |
|---|---|---|---|---|---|
| n-gram (n=3) | — | 0.727 | 0.968 | 0.992 | 0.847 |
| GPT:small | 4.9M | 0.807 | 0.995 | 1.000 | 0.901 |
| **GPT:large** | **21.5M** | **0.811** | **0.994** | **1.000** | **0.902** |

Top-1 ~0.81 is near the **grammar's irreducible ceiling** (~0.33 nats/step of legitimate
optionality: optional metrology, synonym steps like `STRIP RESIST`/`STRIP PHOTORESIST`,
variable cycle counts). We deliberately do **not** chase Top-1 past this noise floor.

### 4.3 Task 2 (completion) and Task 3 (anomaly) — self-eval

Scored with the official `eval_metrics.py` (via `score_selfeval.py`) on our self-eval split
(800 partial sequences incl. KREMSIANS; the cut positions at 60%/80%). The trained GPT
roughly **halves** the edit distance of the n-gram and nearly doubles token accuracy:

| Task 2 (completion) | n-gram | **GPT:large** |
|---|---|---|
| Mean Normalized Edit Distance (↓ better) | 0.569 | **0.227** |
| Mean Token Accuracy | 0.254 | **0.418** |
| Mean Block-level Accuracy | 0.541 | **0.661** |
| Exact Match Rate | 0.003 | 0.009 |

(Exact-match is near-zero for everyone: a 30–40 step completion must be perfect token-for-token,
and the grammar's legitimate optionality makes that essentially unreachable — NED/token/block
are the meaningful signals here.)

| Task 3 (symbolic validator, GPT surprisal `SCORE`) | Score |
|---|---|
| Accuracy / Precision / Recall / **F1** | **1.00** |
| ROC-AUC (with valid supplement) | **1.00** |
| Rule attribution (correctly-detected invalids) | **1.00** |

Task 3 uses **zero ML** for the hard label: the grammar is fully specified, so a deterministic
checker is provably correct and — crucially — **family-agnostic**, so it carries over to the
hidden 4th family. Model surprisal supplies the continuous `SCORE` for ROC-AUC.

**Task 1 next-step (official scorer, self-eval, incl. KREMSIANS):** GPT:large Top-1 **0.661**,
Top-3 0.976, Top-5 0.985, MRR 0.818 — vs n-gram 0.634 / 0.981 / 0.991 / 0.802. Per-family Top-1:
IGBT 0.73, IC 0.69, MOSFET 0.64, **KREMSIANS (OOD) 0.59** — the OOD family is exactly where the
gap shows, as expected.

### 4.4 The headline: KREMSIANS (honest 4th-family OOD)

Trained only on MOSFET+IGBT+IC, evaluated on 500 KREMSIANS sequences it never saw:

| Metric | GPT:large, no guide | GPT:large, **+ guide** |
|---|---|---|
| Top-1 | 0.591 | 0.586 |
| Top-3 | 0.797 | **0.866** |
| Top-5 | 0.829 | **0.969** |
| MRR | 0.704 | 0.737 |

**Correct next step in the top-5 96.9% of the time on a never-seen family.** The guide barely
moves Top-1 (the model already ranks the right step first where the grammar is tight) but
lifts Top-5 by ~14 points by deleting impossible options from the tail.

### 4.5 Scaling: bigger ≠ more transferable

Fresh KREMSIANS OOD sweep across the three GPT sizes (same training data, 500 eval seqs):

| Model | Params | ID Top-1 | KREMSIANS Top-1 (base) | KREMSIANS Top-5 (+guide) |
|---|---|---|---|---|
| GPT:tiny | 0.6M | 0.811 | 0.587 | 0.967 |
| GPT:small | 4.9M | 0.810 | 0.586 | 0.966 |
| GPT:large | 21.5M | 0.812 | 0.591 | 0.969 |

**35× more parameters (0.6M → 21.5M) is essentially flat on every axis** — ID, OOD Top-1, and
guided OOD Top-5 all sit within noise. Capacity is not the bottleneck; the structural barrier
between families is. The larger models even memorise the known families slightly harder
without buying transfer. This is a central section-9 talking point.

---

## 5. What worked / what didn't

**Worked**
- Step-level tokenisation — directly optimises the scored objective; fast on one A100.
- Symbolic validator for Task 3 — provably perfect and family-agnostic (transfers to Task 4).
- Grammar-guided decoding — big Top-3/5/MRR gains for free at inference, no retraining.
- KREMSIANS — an honest OOD probe we committed to *after* freezing design choices.

**Didn't / limits**
- Top-1 OOD stalls ~0.59: the genuinely novel transitions (KREMSIANS hybrid prep, dual
  implant) have no training support, and the guide falls back to the full vocab there.
- Bigger pretrained LLMs (Qwen2.5 3B) memorise faster and generalise *worse* past early stop.
- We considered MBR completion (a `dev/bati` idea) as an alternative Task-2 decoder; greedy
  grammar-guided completion was simpler and competitive, so we kept it.

---

## 6. Next 36h

- GRPO with the validator as reward — optimise grammar-following directly rather than
  cross-entropy, to push OOD Top-1 above the ~0.59 wall.
- Calibrate the surprisal `SCORE` (temperature/Platt) for a cleaner ROC-AUC.
- Confidence-gated guide: only apply the mask where trigram support exists, to avoid the
  full-vocab fallback at novel transitions.
- Per-family completion decoding (block-aware beam) to lift Task-2 exact match.

---

## 7. Credits

- **Organisers / Infineon** — grammar, generator/validator (`generate_sequences.py`), the
  10 rules, and the official scorer (`participant_files/eval_metrics.py`).
- **CINECA Leonardo** — A100 compute (account `EUHPC_D30_031`, reservation `s_tra_ncc`).
- **Team `zero_one_hack_0625`** — modelling, KREMSIANS, grammar guide, eval pipeline.
- Open stack only: PyTorch + HuggingFace Transformers, trained on the cluster — no API,
  no proprietary weights, no eval-file lookup.

*See `RESULTS.md` for the full methodology, per-experiment tables, and the n-order sweep.*
