# CLAUDE.md — Agent guide for the Industrial AI track

This repo trains and benchmarks models that learn semiconductor **process logic**.
`PLAN.md` is the source of truth — read it before writing code. This file tells you
*how* to work in the repo.

## What we are doing (one paragraph)
Sequences of ~115–151 process steps (3 families: MOSFET, IGBT, IC; shared ~120-step
vocabulary). We train a from-scratch decoder-only transformer and a non-neural n-gram
baseline, then benchmark three tasks (next-step prediction, sequence completion, anomaly
detection) on the organizers' fixed eval files, and we optimize for generalization to a
held-out family (proxy for the hidden Task 4).

## Ground rules
- **Read `training_data/generation_rules.md` before touching modeling or anomaly code.**
  §3 lists the 10 forbidden patterns; §5 defines the exact submission format. Match §5 exactly.
- **Never invent the submission format or rule definitions** — read them from the repo.
- **Build against the shared interfaces in `PLAN.md §3`** (`next_step_ranking`, `complete`,
  `sequence_surprisal`). Models must be swappable in `eval_runner.py`.
- **Train from random init** (`GPT2LMHeadModel(cfg)`); do NOT download pretrained weights —
  the cluster may be offline and the judges reward an open, non-API stack.
- **Determinism:** seed everything with `42`. Vocab is built with `sorted()` for stable ids.
- **W&B:** every run logs to project `industrial-ai` with the schema in `PLAN.md §10`.
  Use `WANDB_MODE=offline` on air-gapped nodes, then `wandb sync`.

## Already done (do not rewrite, import these)
- `src/tokenizer.py` — `StepTokenizer.build_from_sequences()`, `.encode/.decode/.save/.load`
- `src/data.py` — `load_all()`, `train_val_split()`, `leave_one_family_out(holdout)`
- `src/models/ngram.py` — `NGramModel(n,alpha).fit().next_step_ranking()/.complete()/.sequence_surprisal()`

## Build next (priority order — see PLAN.md §1 for the full tree)
1. `src/models/gpt.py` (use the `build_gpt` snippet in PLAN.md §4.2; add the shared interface)
2. `src/train.py` (HF `Trainer`, `report_to="wandb"`, config-driven, supports `--leave-out`)
3. `src/eval_runner.py` + `src/infer.py` + `src/blocks.py` (produce §5-format CSVs)
4. `scripts/01_make_anomaly_trainset.py` + `src/anomaly.py` (validate with `generate_sequences.py --validate`)
5. `src/viz.py` + `dashboard/app.py`

## Commands you will use
```bash
python -m src.train --config configs/gpt_small.yaml --train-families mosfet igbt ic --run-name gpt_small_allfam
python -m src.train --config configs/gpt_small.yaml --leave-out mosfet --run-name gpt_small_loo_mosfet
python -m src.eval_runner --task anomaly --model outputs/gpt_small --input eval_input_anomaly.csv --out submissions/task3_gpt.csv
python eval_metrics.py --task anomaly --ground-truth self_eval/anomaly_gt.csv --predictions submissions/task3_gpt.csv
```

## Definition of done for any task module
- Runs on the n-gram AND the GPT through the shared interface.
- Produces a CSV that passes `eval_metrics.py` against a `self_eval/` ground-truth file.
- Logs its metrics to W&B under the `PLAN.md §10` keys.
- Has a one-line `if __name__ == "__main__":` smoke test.

## Do not
- Do not break a frozen submission in `submissions/`; copy and improve instead.
- Do not add a family-embedding token to the model (hurts OOD — see PLAN.md §4.2).
- Do not commit `outputs/` (checkpoints/logs) — keep it gitignored.
