# PLAN.md — Industrial AI Track: Learning & Benchmarking Process Logic

> **Winning thesis (read this first, repeat it in the pitch):**
> We are not building a next-step predictor. We are building **evidence that a model
> learns the process grammar instead of memorizing surface patterns.** Four
> independent pieces of evidence: (1) it generalizes to a family it never saw,
> (2) it rediscovers the 10 forbidden-pattern rules without being told them,
> (3) its per-step surprisal pinpoints exactly where a rule is violated, and
> (4) that understanding scales predictably with data and model size.

This file is the single source of truth. It is written so **Claude Code can expand
each module from the specs below** and so **four people can work in parallel without
blocking each other.** Read `CLAUDE.md` for agent conventions.

---

## 0. The one strategic edge

The scored-but-not-submitted **Task 4 (OOD on a hidden 4th family)** is the heart of
the challenge, and most teams will ignore it because it is not in the submission table.
The README confirms the key fact that makes it attackable: **vocabulary is shared
across families (~120 steps); families differ only in which optional blocks appear and
in cycle counts.** So the hidden family uses the same words in a new grammar.

Therefore **`leave_one_family_out()` is a near-perfect local proxy for Task 4.** Every
design decision is judged on the held-out family, not just on in-distribution accuracy.
This single discipline is our differentiator; everything below serves it.

---

## 1. Repository structure

```
.
├── PLAN.md                       ← this playbook (source of truth)
├── CLAUDE.md                     ← instructions for Claude Code agents
├── README.md                     ← (existing) repo readme
├── Track_industrial_en.md        ← (existing) full briefing
├── training_data/                ← (existing) data + generator + rules
│   ├── generation_rules.md       ←  §1 vocab/12 categories · §2 grammar ·
│   │                                §3 the 10 forbidden patterns · §4 11 variation
│   │                                axes · §5 eval protocol · §6 CLI  ← MINE THIS
│   ├── generate_sequences.py     ←  --family --count --output --seed | --validate | --estimate-only
│   ├── *_variants.csv            ←  1000 seqs/family, long format (SEQUENCE_ID, STEP)
│   ├── synthetic*.csv            ←  1 canonical reference per family
│   └── *_Longdescr*.csv          ←  enriched (descriptions + fab params) — optional
├── src/
│   ├── tokenizer.py              ← ✅ DONE  step-level tokenizer (vocab built from data)
│   ├── data.py                   ← ✅ DONE  loaders + train/val + leave_one_family_out
│   ├── blocks.py                 ← TODO   step→block map (12 categories) for block-acc + probes
│   ├── models/
│   │   ├── ngram.py              ← ✅ DONE  stupid-backoff n-gram baseline
│   │   └── gpt.py                ← TODO   HF GPT-2 (from scratch) wrapper, shared interface
│   ├── train.py                  ← TODO   HF Trainer loop + W&B
│   ├── infer.py                  ← TODO   next-step + completion generation
│   ├── anomaly.py                ← TODO   surprisal scoring + rule classifier + validator oracle
│   ├── eval_runner.py            ← TODO   build submission CSVs, call eval_metrics.py, log to W&B
│   └── viz.py                    ← TODO   embeddings UMAP, surprisal plots, scaling/OOD charts
├── configs/
│   ├── gpt_tiny.yaml  gpt_small.yaml  gpt_large.yaml   ← TODO model sizes
│   └── sweep_scaling.yaml        ← TODO W&B sweep (size × data volume)
├── scripts/
│   ├── 00_setup.sh               ← TODO env + wandb login + smoke test
│   ├── 01_make_anomaly_trainset.py ← TODO inject the 10 rules → labeled anomalies
│   ├── 02_build_self_eval.py      ← TODO held-out labeled set for offline scoring
│   └── slurm_train.sh            ← TODO Leonardo SLURM submission stub
├── submissions/                  ← output CSVs we hand in
├── self_eval/                    ← our own ground-truth files for offline scoring
├── dashboard/app.py              ← TODO Streamlit (bonus, explicitly rewarded)
├── notebooks/                    ← exploration + the live demo
└── outputs/                      ← checkpoints, logs, artifacts (gitignored)
```

Modules marked ✅ DONE are committed and tested. Everything else is fully specified below.

---

## 2. Setup (`scripts/00_setup.sh`)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers wandb scikit-learn umap-learn matplotlib pyyaml pandas streamlit
wandb login                       # on Leonardo without internet use: export WANDB_MODE=offline
# smoke test
python src/tokenizer.py
python -c "import sys;sys.path.insert(0,'src');from models.ngram import NGramModel;print('ok')"
```

**Cluster note (Leonardo):** the GPT models train from *random init* (`GPT2LMHeadModel(cfg)`),
so **no Hugging Face Hub download is needed** — works offline and fits the "open stack, not
an API wrapper" ethos the judges reward. If nodes are air-gapped, run `WANDB_MODE=offline`
and `wandb sync` from the login node afterwards.

---

## 3. Shared interfaces (define these first, then everyone parallelizes)

All models expose the **same three methods** so `infer.py`, `anomaly.py` and `eval_runner.py`
are model-agnostic and swappable between the n-gram and the GPT:

```python
model.next_step_ranking(prefix: list[str], k: int) -> list[str]   # Task 1
model.complete(prefix: list[str]) -> list[str]                    # Task 2 (roll-out to <eos>)
model.sequence_surprisal(seq: list[str]) -> float                 # Task 3 (anomaly score)
# GPT additionally exposes:
model.per_step_surprisal(seq: list[str]) -> list[float]           # for the localization plot
model.step_embeddings() -> dict[str, np.ndarray]                  # for UMAP / probes
```

`src/data.py` already provides `load_all()`, `train_val_split()`, `leave_one_family_out()`.
`src/tokenizer.py` already provides `StepTokenizer.build_from_sequences()`. Build everything
against these signatures so stubs can be filled independently.

---

## 4. The models — exact specs

### 4.1 Baseline — n-gram (DONE: `src/models/ngram.py`)
Trigram with stupid backoff (`n=3, alpha=0.4`). Tune `n∈{2,3,4}`, `alpha∈[0.3,0.7]` on the
val split. This is the honest bar — on structured sequences it is strong, so **the GPT must
clearly beat it on the held-out family, not just in-distribution.** That gap *is* the story.

### 4.2 Hero — decoder-only transformer (from scratch), `src/models/gpt.py`
Use **Hugging Face `GPT2LMHeadModel` with a custom tiny `GPT2Config` and our step-level
tokenizer.** Rationale: HF `Trainer` gives W&B logging, checkpointing and `generate()` for
free; random init means no download; fully open and reproducible.

```python
from transformers import GPT2Config, GPT2LMHeadModel

def build_gpt(tok, n_embd=256, n_layer=6, n_head=8, n_positions=256, dropout=0.1):
    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=n_positions, n_ctx=n_positions,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        bos_token_id=tok.bos_id, eos_token_id=tok.eos_id, pad_token_id=tok.pad_id,
        resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
    )
    return GPT2LMHeadModel(cfg)
```

**Size ladder (for the scaling study):**

| config       | n_embd | n_layer | n_head | ~params | role                                   |
| ------------ | ------ | ------- | ------ | ------- | -------------------------------------- |
| `gpt_tiny`   | 128    | 2       | 4      | ~0.5–1M | fast iteration + 100-seq scaling point |
| `gpt_small`  | 256    | 6       | 8      | ~6M     | **default hero model**                 |
| `gpt_large`  | 384    | 12      | 12     | ~25M    | top of the scaling curve               |

**Critical OOD design decision — DO NOT use a family-embedding token.** A model conditioned
on a family id cannot use it on the hidden family. Let the model **infer the regime from the
prefix** (the first ~10 steps already reveal the family). Run both as an ablation and report
the held-out gap — the "inferred-regime beats hard-coded-family on OOD" result is a strong slide.

Max sequence length is 151 → `n_positions=256` is comfortable. Pad/truncate to longest in batch.

---

## 5. Training (`src/train.py`)

HF `Trainer` with `report_to="wandb"`. Objective: standard causal LM (next-step) cross-entropy.

**Commands (the canonical runs):**
```bash
# hero model, all three families
python -m src.train --config configs/gpt_small.yaml \
  --train-families mosfet igbt ic --val-frac 0.1 \
  --epochs 10 --batch-size 64 --lr 3e-4 \
  --wandb-project industrial-ai --run-name gpt_small_allfam --out outputs/gpt_small

# leave-one-family-out (run for EACH family) — our Task-4 proxy
python -m src.train --config configs/gpt_small.yaml \
  --leave-out mosfet --run-name gpt_small_loo_mosfet --out outputs/loo_mosfet
```

**Default hyperparameters:** AdamW, lr `3e-4`, cosine schedule, 5% warmup, weight decay `0.01`,
batch `64`, `10` epochs, eval every `0.25` epoch, save best on `eval/loss`, grad clip `1.0`,
mixed precision (`bf16` on Leonardo), seed `42`.

**`configs/gpt_small.yaml` (example):**
```yaml
model: {n_embd: 256, n_layer: 6, n_head: 8, n_positions: 256, dropout: 0.1}
train: {epochs: 10, batch_size: 64, lr: 3.0e-4, warmup_ratio: 0.05,
        weight_decay: 0.01, eval_steps_frac: 0.25, seed: 42, bf16: true}
data:  {train_families: [mosfet, igbt, ic], val_frac: 0.1, max_len: 256}
```

---

## 6. The three submission tasks (`src/infer.py`, `src/eval_runner.py`)

Read `generation_rules.md §5` for the exact submission file format and column names, then
make `eval_runner.py` emit precisely that. **Lock a valid dummy submission through
`eval_metrics.py` in the first 3 hours** — format bugs at hour 35 lose hackathons.

| # | Task              | Model call                          | Output → metric (via `eval_metrics.py`)                         |
|---|-------------------|-------------------------------------|-----------------------------------------------------------------|
| 1 | Next-step pred.   | `next_step_ranking(prefix, k=5)`    | Top-1/3/5 Accuracy, MRR                                          |
| 2 | Completion        | `complete(prefix)` (greedy; try beam)| Exact Match, Normalized Edit Distance, Token Acc, Block Acc     |
| 3 | Anomaly detection | `sequence_surprisal(seq)` + classifier | Binary Acc, P, R, F1, ROC-AUC, Rule Attribution Acc, Confusion |

**Submission/eval commands:**
```bash
python -m src.eval_runner --task next_step  --model outputs/gpt_small --input eval_input_valid.csv   --out submissions/task1_gpt.csv  --wandb
python -m src.eval_runner --task completion --model outputs/gpt_small --input eval_input_valid.csv   --out submissions/task2_gpt.csv  --wandb
python -m src.eval_runner --task anomaly    --model outputs/gpt_small --input eval_input_anomaly.csv --out submissions/task3_gpt.csv  --wandb
# self-score against OUR held-out ground truth (organizer files have none):
python eval_metrics.py --task anomaly --ground-truth self_eval/anomaly_gt.csv --predictions submissions/task3_gpt.csv
```

**Block accuracy** needs `src/blocks.py`: a `step → block` map for the 12 backbone categories
(Logistics → Clean → Family Prep → Oxidation → Litho/Etch/Implant → ILD → Via → Metal →
Passivation → Backside → Final Inspection → Test → Ship). Build it from `generation_rules.md §1`.

---

## 7. Anomaly pipeline (Task 3) — the "rediscover the rules" angle

We already own a perfect symbolic checker: `generate_sequences.py --validate` tests all 10
forbidden patterns. Use it three ways:

1. **Labeled training data (`scripts/01_make_anomaly_trainset.py`):** take valid sequences,
   write one *injector* per rule in `generation_rules.md §3` (e.g. delete the clean before a
   deposition; move an electrical test before passivation), label each row with its rule id,
   and confirm with `--validate` that injected ones fail and originals pass.
2. **Neural detector (the scored submission):** rank sequences by GPT `sequence_surprisal`;
   threshold on the val split for Binary Acc/F1, use the raw score for ROC-AUC. For **rule
   attribution**, train a small classifier (10 rule classes) on the labeled set, using
   per-step surprisal + GPT hidden states as features.
3. **Symbolic oracle (the narrative):** the validator is the upper bound. **Headline metric:
   "% of the validator's detections our trained model recovers with no rule supervision."**
   That number is the cleanest possible evidence of learned logic.

**Signature visualizations (memorable, cheap):**
- **Surprisal localization:** plot `per_step_surprisal(seq)` for an injected anomaly — the
  spike lands exactly on the violating step. One picture = proof.
- **Counterfactual probes:** remove a required clean before a deposition and measure the
  probability drop on the deposition. Knows-the-rule = big drop.
- **Rule-recovery curve (creative, do this):** during training, every N steps run the probe
  set and plot *how many of the 10 rules the model has implicitly learned vs training step.*
  Learning dynamics of process logic — nobody else will have this.

---

## 8. Generalization (Task 4 proxy) — `leave_one_family_out`

For each held-out family, train on the other two and evaluate **all of Tasks 1–3** on the
held-out one. Report, per main metric, the **ID → OOD drop**. Optimize design choices
(prefix-inferred regime, dropout, data volume) to minimize this drop. This is the closest
thing to what the organizers will actually measure, and the bar the n-gram cannot clear.

---

## 9. Scaling study (Level 3 / stretch) — `configs/sweep_scaling.yaml`

Grid: **{tiny, small, large}** × **{100, 1000, 5000} training sequences** (generate the 5000
with `generate_sequences.py --count`). Log every cell. Two headline plots:
- main metric **vs #training sequences** (per model size),
- the **OOD drop vs scale** — does more data/size close the held-out-family gap?

```bash
wandb sweep configs/sweep_scaling.yaml   # prints SWEEP_ID
wandb agent <SWEEP_ID>                    # run agents (parallel across GPUs)
```

---

## 10. W&B — what to log (the "initial metrics of interest")

One project `industrial-ai`; tag runs `baseline | hero | loo | scaling`. Log schema:

| group        | keys                                                                              |
| ------------ | --------------------------------------------------------------------------------- |
| training     | `train/loss` `eval/loss` `train/ppl` `eval/ppl` `lr` `grad_norm` `tokens_per_sec` |
| task1        | `task1/top1` `task1/top3` `task1/top5` `task1/mrr` (× overall + per family)        |
| task2        | `task2/exact_match` `task2/norm_edit_dist` `task2/token_acc` `task2/block_acc` (× 60% / 80%) |
| task3        | `task3/acc` `task3/precision` `task3/recall` `task3/f1` `task3/roc_auc` `task3/rule_attr_acc` + `wandb.plot.confusion_matrix` |
| ood (proxy)  | `ood/<metric>_id` `ood/<metric>_ood` `ood/<metric>_drop` (per held-out family)     |
| rules        | `rules/recovered_count` (0–10), `rules/recovery_rate_vs_validator`                 |

**W&B Tables (judges love these):**
- `predictions`: `prefix · truth_next · ngram_pred · gpt_pred` (side-by-side baseline vs hero).
- `anomaly_examples`: `seq · true_label · score · pred_rule · surprisal_curve`.

**Custom charts / artifacts:** scaling line chart (group by config), OOD-drop bar chart,
UMAP of step embeddings (`viz.step_embeddings → umap`), surprisal-localization images,
rule-recovery curve.

**What "good" roughly looks like (intuitions, not promises):** n-gram Top-1 likely high
(~70–90%) thanks to strong structure; the GPT should *match it in-distribution but lose
much less on the held-out family* — that delta is the win. Exact-match completion will be
low; lean on edit distance and block accuracy. Anomaly ROC-AUC should be high; the
interesting number is rule-recovery vs the validator.

---

## 11. Differentiators, ranked (what actually wins)

1. **Leave-one-family-out OOD proxy** — must. Targets the scored Task 4 nobody optimizes.
2. **"Can the model rediscover the rules?"** — must, cheap. Neural surprisal vs symbolic oracle.
3. **Scaling study with OOD-drop-vs-scale** — high. The track's central research question, plotted.
4. **Surprisal localization + counterfactual probes + rule-recovery curve** — cheap, memorable.
5. **Step-embedding UMAP colored by the 12 categories** — pretty proof of learned semantics.
6. **Streamlit dashboard** — explicitly rewarded; ties it all together for the demo.

---

## 12. Four-person division (parallel, interface-first)

Hour 0–1 everyone reads `generation_rules.md §1–§5` together. Then split. The only hard
dependency is `tokenizer.py` + `data.py` (already DONE), so all four start at once.

| Person | Role                         | Owns                                                              | First deliverable (by ~h6)                        |
| ------ | ---------------------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| **A**  | Modeling & scaling           | `models/gpt.py`, `train.py`, `configs/*`, `sweep_scaling.yaml`    | hero `gpt_small` training + W&B loss curves       |
| **B**  | Eval, tasks 1&2, submissions | `infer.py`, `eval_runner.py`, `blocks.py`, `02_build_self_eval.py`, n-gram tuning | **valid dummy submission through `eval_metrics.py`** + n-gram baseline numbers |
| **C**  | Anomaly & rules (Task 3)     | `01_make_anomaly_trainset.py`, `anomaly.py`, validator oracle, probes | labeled anomaly set + n-gram surprisal ROC-AUC    |
| **D**  | Generalization, viz, story   | LOO experiments, `viz.py`, `dashboard/app.py`, slides, demo notebook | LOO harness running + first OOD-drop bar chart    |

Dependencies & handoffs: B's `eval_runner` consumes A's checkpoints and C's anomaly scores
via the shared interface in §3 — so B builds against the n-gram first and swaps in the GPT
when A's checkpoint lands. D consumes everyone's W&B runs; D never blocks, always polishing.
Anyone idle picks up the next `TODO` in §1.

---

## 13. Hour-by-hour timeline (36h)

| Window  | Goal                                                                                          |
| ------- | --------------------------------------------------------------------------------------------- |
| 0–3h    | Read rules together. B: dummy submission validates through `eval_metrics.py`. A: `gpt.py` skeleton runs one step. C: first injectors. D: LOO harness. |
| 3–6h    | n-gram baseline numbers banked (B+C). First real GPT run training (A). **First real submission saved.** |
| 6–16h   | Tune hero model (A). Anomaly detector + rule classifier + oracle comparison (C). Tasks 1&2 polished, block-acc (B). LOO runs for all 3 families (D+A). Bank a strong full submission. |
| 16–26h  | Scaling sweep (A). Counterfactual + rule-recovery curve (C). OOD-drop charts + embedding UMAP (D). |
| 26–32h  | Streamlit dashboard + demo notebook (D). Lock best submission, freeze it (B). |
| 32–36h  | Slides, rehearse pitch, buffer, sleep rotation. **Never break the frozen submission — only improve a copy.** |

---

## 14. Submission checklist (freeze before the deadline)

- [ ] `submissions/task1_gpt.csv`, `task2_gpt.csv`, `task3_gpt.csv` in the exact `§5` format
- [ ] all three pass `eval_metrics.py` on our `self_eval/` ground truth with sane numbers
- [ ] baseline (n-gram) vs hero vs LOO numbers in one W&B report
- [ ] reproducible: `seed=42`, configs committed, `README` with exact run commands
- [ ] dashboard or demo notebook shows baseline-vs-hero on identical inputs
- [ ] one-line pitch rehearsed (top of this file)

---

## 15. The pitch (60 seconds)

> "Every team trained a model that predicts the next process step. We asked a harder
> question: does it *understand* the process or just memorize it? Our model generalizes to
> a product family it has never seen, it rediscovers the ten process-logic rules without
> ever being told them — its surprisal spikes exactly on the violating step — and we show
> that this understanding grows predictably as we scale data and model size. That's learned
> process logic, not surface pattern matching."
