# Team Kremsians Pitch Talk Track

Target: 4 minutes. The deck can be presented faster if the jury cuts questions
early, but this is the intended full spoken story.

## Slide 1 - Opening

Team Kremsians built a process-logic system for the Industrial AI Infineon
challenge. The core claim is simple: we did not only generate the three CSVs;
we built a benchmarked sequence model that learns semiconductor process order.

## Slide 2 - Challenge

The challenge asks for next-step prediction, sequence completion, and anomaly
detection from partial process traces. The important design choice is to use
learning for probabilistic sequence continuation, but use explicit rule logic
where the challenge already gives process rules.

## Slide 3 - Model

Our model is a GPT decoder trained from scratch. It is not an API wrapper and
not a pretrained language model. Each fab step is a single token, so the model
is trained directly on the unit that the scorer evaluates. The n-gram baseline
stays in the repo as an honest comparator and fallback.

## Slide 4 - OOD Protocol

We were skeptical of scores that only fit the known families. We therefore use
15 total families: the 3 organizer families plus 12 synthetic process families.
The final protocol trains on 12 families and evaluates on 3 held-out families,
with zero official eval rows used as training data.

## Slide 5 - Results

The trained GPT improves next-step Top-1 and Top-5, and it improves completion
quality on normalized edit distance, token accuracy, and block accuracy. Task 3
is handled by the deterministic process-rule validator, which gives perfect
F1, ROC-AUC, and rule attribution on our self-eval split.

## Slide 6 - Same-Prefix Demo

Here is the behavior difference on one real MOSFET prefix. The n-gram baseline
places the exact next step second. The trained GPT places it first. That is the
story we want the jury to remember: the transformer learns enough context to
resolve a realistic process continuation.

## Slide 7 - Reproducibility

The same repo runs locally on an Apple Silicon MacBook using MPS or on Leonardo
using CUDA. The checkpoint, loss curve, official-scorer self-eval reports,
official-format CSVs, and SLURM job are included.

## Slide 8 - Close

Kremsians built a complete, reproducible benchmark and submission package:
baseline, trained model, out-of-distribution protocol, explicit rules, and all
three required CSVs. The next engineering improvement is validator-guided beam
search for completion.
