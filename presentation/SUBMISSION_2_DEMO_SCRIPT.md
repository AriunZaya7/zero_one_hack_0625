# Team Kremsians Two-Minute Demo Script

This is the spoken demo-video script. It intentionally avoids internal branch
or submission-iteration names.

## 0:00-0:14 Problem

We receive partial semiconductor process sequences and must produce three
official files: `nextstep.csv`, `completion.csv`, and `anomaly.csv`.

## 0:14-0:38 Local Mac Run

Show a realistic local terminal starting the benchmark and training command on
Apple Silicon. Mention that the script auto-detects MPS and that this is a
from-scratch step-token GPT, not an LLM wrapper.

## 0:38-0:56 Leonardo Run

Fade to the same repository running on Leonardo through `sbatch job.slurm`.
Mention CUDA/A100, official CSV generation, and organizer-metric self-eval.

## 0:56-1:18 OOD Protocol

To avoid choosing a model that only memorizes known families, we evaluate with
15 families total: the 3 organizer families plus 12 synthetic process families.
The benchmark trains on 12 and tests on 3 unseen families, then reports the
family-wise average.

## 1:18-1:44 Results

On the held-out families, GPT improves OOD Top-1 from 0.662 to 0.722 and OOD
Top-5 from 0.976 to 0.999. On the official-scorer self-eval split, completion
NED drops from 0.336 to 0.263, and anomaly F1 plus rule attribution are 1.00.

## 1:44-2:00 Final Package

The repo includes the checkpoint, loss curve, score reports, official-format
CSVs, pitch deck, demo video, and the same commands for Apple Silicon MPS or
Leonardo CUDA.
