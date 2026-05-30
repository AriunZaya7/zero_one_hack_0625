# HANDOFF.md — read this first (handoff to the Leonardo Claude Code session)

> **You (Claude Code) are continuing work started on another machine.** This file
> is the bridge. Read it top to bottom, then skim `PLAN.md` (strategy/source of
> truth) and `CLAUDE.md` (conventions). `PLAN.md` describes an *aspirational* tree;
> the **actual layout differs** — see §3 below. Trust this file for current state.

---

## 1. The project in one paragraph

Industrial AI hackathon track. We model semiconductor **process-step sequences**
(3 families: MOSFET, IGBT, IC; ~129 steps each; shared vocab of **198 steps**).
The winning thesis (PLAN.md §0): show a model **learns the process grammar, not
surface patterns** — proven mainly by **generalization to a held-out family**
(`leave_one_family_out`, our proxy for the hidden Task 4). Everything is judged on
the **OOD (held-out) family**, not in-distribution accuracy.

## 2. What is DONE and working (verified on CPU)

| File | What it is | State |
|---|---|---|
| `tokenizer.py` | step-level tokenizer (1 step = 1 token), vocab from data | pre-existing, DONE |
| `data.py` | `load_all`, `train_val_split`, `leave_one_family_out` | pre-existing, DONE |
| `ngram.py` | n-gram baseline (stupid backoff), shared interface | pre-existing, DONE |
| `run_baseline.py` | n-gram next-step eval (top-1/3/5, MRR) ID + all 3 LOO | DONE, ran |
| `ablation_n.py` | n-gram order sweep (n=2..15) ID vs OOD | DONE, ran |
| `gpt.py` | **from-scratch GPT-2** (tiny/small/large) + shared interface | DONE, trained |
| `hf_model.py` | **pretrained Qwen2.5** wrapper, closed-vocab scoring | DONE, path tested |
| `train.py` | unified trainer (gpt OR hf), ID/LOO, per-run JSON results | DONE, ran |
| `viz.py` | plots from `results/*.json` | DONE, ran |
| `scripts/slurm_train.sh` | Leonardo SLURM launcher (offline-safe) | written, needs `--account` |
| `scripts/prefetch_models.py` | login-node HF weight pre-download | written |

### Key results so far (the story)
- **n-gram baseline:** ID next-step top-1 ≈ **0.72**, OOD (held-out family) ≈ **0.44–0.50**.
- **n-order sweep finding (important):** bigger n raises ID (→0.79 at n=10) but OOD
  **plateaus at ~0.48** and the gap *widens*. → pure memorization can't generalize.
  We use **n=3** as the honest baseline.
- **from-scratch GPT (tiny, 0.46M, CPU):** OOD top-1 ≈ **0.50–0.53**, already ≥ n-gram
  and clearly better on top-3/5/MRR. Undertrained — the margin should grow with
  size/epochs/GPU. This is the thesis in miniature.

## 3. Layout reality (do NOT follow PLAN.md's tree blindly)
- All modules are at the **repo root**, NOT in `src/`. Import flat: `from data import ...`.
- **`eval_metrics.py` and the organizer eval-input CSVs are NOT in this checkout.**
  PLAN.md §6 assumes them. **Locate them before building `eval_runner.py` / submissions.**
- Data: `training_data/{MOSFET,IGBT,IC}_variants.csv` (cols `SEQUENCE_ID,STEP`,
  1000 seqs/family). Rules: `training_data/generation_rules.md` (§3 = 10 forbidden
  patterns, §5 = submission format). `generate_sequences.py --validate` is the
  symbolic anomaly oracle.

## 3b. Teammate work already in the repo (coordinate, don't clobber)
This is a **shared team repo** (`git@github.com:AriunZaya7/...`, branches `main`,
`dev/bati`, `morgan`). Besides our root modules, `main` already contains parallel
efforts — be aware so we reuse rather than duplicate:
- **`small_transformer_model/`** — a teammate's from-scratch transformer + **working
  Leonardo setup**: `train/setup_leonardo.sh`, `train/job.slurm`, `train/train_model.py`,
  prepared `data/*.jsonl` + `vocab.json`. Our `scripts/slurm_train.sh` borrows their
  proven values: modules `python/3.10` + `cuda/12.1`, partition `boost_usr_prod`, venv
  at `~/venv`, torch via the `cu121` wheel index. **The CINECA `--account` is a
  placeholder in both their job.slurm and ours — get the real one from the team.**
  Note their setup targets **Qwen2-0.5B**; we standardized on the newer **Qwen2.5**
  ladder for the scaling curve.
- **`test_baseline_1/`**, **`test_baseline_2/`** — teammates' n-gram / LLM / random
  baselines with W&B logs (their `wandb/` dirs are committed; ours are gitignored).

Our line of work (root `gpt.py`/`hf_model.py`/`train.py`) is the **two-tier comparison**
(from-scratch GPT vs fine-tuned Qwen2.5, unified closed-vocab eval). It overlaps
`small_transformer_model/` conceptually — **check with the team before merging to main**;
for now it lives on the `cristi/modeling` branch.

## 4. The shared interface (every model implements this — keep it)
```python
model.next_step_ranking(prefix: list[str], k) -> list[str]   # Task 1
model.complete(prefix) -> list[str]                          # Task 2
model.sequence_surprisal(seq) -> float                       # Task 3
# GPT/HF extras: per_step_surprisal(seq), step_embeddings()
```
Both tiers are scored **identically** via **closed-vocab next-step ranking**: rank
the 198 known steps by the model's log-prob. (Lets a subword Qwen be compared to the
step-level GPT and n-gram apples-to-apples.) See `hf_model.py:_score_candidates`.

## 5. Decisions already made (don't relitigate without reason)
- **Two model tiers:** from-scratch GPT-2 (clean "learned-it" anchor) + fine-tuned
  **Qwen2.5 ladder (0.5B/1.5B/3B, maybe 7B)** (performance/transfer + scaling curve).
- **NO family-embedding token** (PLAN.md §4.2 — it kills OOD; the model must infer
  the family from the prefix).
- **Tokenizer built from ALL families** (vocab is shared, incl. the hidden one) so the
  held-out family maps to known ids — a fair OOD test, not artificial `<unk>` noise.
- **Per-run JSON results** in `results/` (one file per run) → safe for parallel SLURM
  jobs (concurrent CSV appends corrupt the file — we hit this and fixed it).
- n-gram n=3, seed=42 everywhere.

## 6. How to run on Leonardo
```bash
# --- LOGIN node (has internet): pre-stage weights once ---
export HF_HOME=$WORK/hf_cache
python scripts/prefetch_models.py            # Qwen2.5 0.5B/1.5B/3B

# --- edit scripts/slurm_train.sh: set  #SBATCH --account=<your_project> ---
# adjust  module load  lines to Leonardo's exact module names; ensure .venv exists

# --- submit jobs (one per model × mode). Args: MODEL  MODE_FLAGS  EPOCHS ---
sbatch scripts/slurm_train.sh "gpt:small"            "--leave-out mosfet" 10
sbatch scripts/slurm_train.sh "hf:Qwen/Qwen2.5-0.5B" "--leave-out mosfet" 3
sbatch scripts/slurm_train.sh "hf:Qwen/Qwen2.5-1.5B" "--leave-out mosfet" 3
# ... repeat for holdouts igbt, ic, AND the ID runs (--train-families mosfet igbt ic)

# --- after jobs finish: build the comparison + scaling plots ---
python viz.py                                # plots/ from results/*.json
```
Notes: GPU is auto-detected. Default lr: gpt `3e-4`, hf `1e-5` (full fine-tune; LoRA
only needed at 7B+). On GPU, **raise `EVAL_SEQS_HF` (default 60)** in `train.py` for
final numbers — closed-vocab eval is much faster on GPU. Air-gapped nodes: the slurm
script already sets `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline`.

## 7. TODO — what's left (priority order)
1. **Fill the experiment grid** on GPU: from-scratch (tiny/small/large) + Qwen2.5
   (0.5B/1.5B/3B) × {3 LOO holdouts + ID}. This produces the OOD-vs-n-gram,
   ID→OOD-drop, and **two scaling curves** (`viz.py`). This is the core evidence.
2. **Locate `eval_metrics.py` + organizer eval CSVs**, then build `eval_runner.py`
   + `infer.py` to emit the §5 submission format. Lock a valid dummy submission early.
3. **`blocks.py`** — step→block map for the 12 categories (from generation_rules §1)
   → block accuracy for Task 2.
4. **Anomaly pipeline (Task 3):** `scripts/01_make_anomaly_trainset.py` (inject the 10
   rules, validate with `generate_sequences.py --validate`), `anomaly.py` (surprisal
   ROC-AUC + rule-attribution classifier). Headline: "% of validator detections our
   model recovers with no rule supervision."
5. **Graduate `train.py` to HF `Trainer` + W&B** (currently a minimal manual loop, no
   W&B). PLAN.md §5/§10 has the config + logging schema. Optional but rewarded.
6. Stretch: `dashboard/app.py` (Streamlit), UMAP of step embeddings, surprisal-
   localization plots, rule-recovery curve.

## 8. Gotchas
- Run from the repo root (flat imports). Seed is 42.
- `outputs/`, `plots/`, `results/`, `.venv/`, `__pycache__/` are gitignored — code is
  tracked, artifacts regenerate. (So expect empty `results/` on a fresh pull — re-run.)
- The from-scratch GPT uses the **step-level** tokenizer; Qwen uses its **native BPE**
  + the closed-vocab scorer. Don't mix them.
- env said "not a git repo" for the parent dir, but the project IS a git repo.
