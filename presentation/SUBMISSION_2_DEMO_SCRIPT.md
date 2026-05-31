# Team Kremsians Two-Minute Demo Script

This is the spoken demo-video script. It intentionally avoids internal branch
or submission-iteration names.

## 0:00-0:15 Problem

We receive partial semiconductor process sequences and must produce three
official files: `nextstep.csv`, `completion.csv`, and `anomaly.csv`.

## 0:15-0:40 Model

For next-step prediction and completion, we trained a GPT decoder from scratch
on process-step tokens. This is not a generic LLM wrapper: one fab step is one
token, and the model is optimized directly for the unit the scorer uses.

## 0:40-1:05 Same-Prefix Example

On a real MOSFET prefix, the n-gram baseline ranks `STRIP PHOTORESIST` second.
The trained GPT ranks the exact ground-truth next step first on the same input.

## 1:05-1:28 OOD Protocol

To avoid choosing a model that only memorizes known families, we evaluate with
15 families total: the 3 organizer families plus 12 synthetic process families.
The benchmark trains on 12 and tests on 3 unseen families, then reports the
family-wise average.

## 1:28-1:50 Results

On the held-out families, GPT improves OOD Top-1 from 0.662 to 0.722 and OOD
Top-5 from 0.976 to 0.999. On the official-scorer self-eval split, completion
NED drops from 0.336 to 0.263, and anomaly F1 plus rule attribution are 1.00.

## 1:50-2:00 Reproducibility

The repo includes the checkpoint, loss curve, score reports, official-format
CSVs, and the same commands for Apple Silicon MPS or Leonardo CUDA.
