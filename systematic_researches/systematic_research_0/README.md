# Systematic Research 0: Industrial AI Process-Sequence Modeling

This research pass asks:

> What reinforcement learning, machine learning, statistics, process-mining,
> and hybrid methods best fit the Industrial AI track problem?

## Short Conclusion

The best immediate method for this repo is **not pure RL**. The task is mostly
offline supervised sequence prediction, suffix prediction, and conformance
checking over structured process traces. There is no live environment to control,
no interactive reward, and no reason to spend scarce hackathon time making RL
stable before a strong supervised baseline exists.

The best practical next step is a **hybrid process-sequence system**:

1. **Context retrieval / nearest historical continuation** for Task 2 sequence
   completion, because the official inputs include 60% and 80% partial traces
   and the generated routes are highly structured.
2. **Count-based transition statistics** as a robust fallback for Task 1
   next-step ranking.
3. **Symbolic process-rule validation** for Task 3 anomaly detection and for
   future constrained decoding.
4. **Step-level transformer from scratch** as the next major model after this
   retrieval hybrid, evaluated with leave-one-family-out as the local proxy for
   hidden OOD Task 4.

This led to:

- `solutions/solution_1_hybrid_retrieval/`

## Research Method

The literature search focused on comparable problems:

- Predictive process monitoring: next activity, suffix prediction, remaining
  time, process outcome.
- Process mining and conformance checking: checking event traces against
  process constraints.
- Log anomaly detection: predicting the next log/event key and flagging unlikely
  or invalid sequences.
- Retrieval-augmented sequence modeling: nearest-neighbor memory for language
  models and structured sequences.
- Constrained decoding: using formal constraints or parsers to reject invalid
  generated outputs.
- Reinforcement learning for sequence generation: sequence-level objectives and
  exposure-bias mitigation.

## Key Sources Reviewed

| Area | Source | Why it matters here |
| --- | --- | --- |
| Process mining | [Process Mining Manifesto, IEEE Task Force](https://www.win.tue.nl/ieeetfpm/lib/exe/fetch.php?media=shared:process_mining_manifesto.pdf) | Frames event logs and process traces as first-class objects for discovery, conformance, and enhancement. |
| Conformance checking | [Rozinat and van der Aalst, Conformance Checking of Processes Based on Monitoring Real Behavior](https://doi.org/10.1016/j.is.2007.07.001) | Task 3 is a conformance problem: does the observed route violate process logic? |
| Predictive process monitoring | [Tax et al., Predictive Business Process Monitoring with LSTM Neural Networks](https://arxiv.org/abs/1612.02130) | Direct analogue for next activity and suffix prediction from partial process traces. |
| Deep process prediction | [Evermann, Rehse, Fettke, Predicting Process Behaviour using Deep Learning](https://doi.org/10.1016/j.dss.2016.12.003) | Shows process behavior can be modeled as sequence learning over event logs. |
| Predictive monitoring benchmark | [Teinemaa et al., Outcome-Oriented Predictive Process Monitoring](https://dl.acm.org/doi/10.1145/3301300) | Important warning: evaluation protocol and baselines matter as much as model choice. |
| Transformers | [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Supports the next major step: a step-level transformer for longer-range dependencies. |
| Process transformers | [ProcessTransformer: Predictive Business Process Monitoring with Transformer Network](https://arxiv.org/abs/2104.00721) | Directly supports using transformer architectures for predictive process monitoring. |
| Log anomaly detection | [Du et al., DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf) | Similar to Task 3: learn normal event/log sequences, use next-event prediction and anomaly signals. |
| Retrieval-augmented LM | [Khandelwal et al., Generalization through Memorization: kNN Language Models](https://arxiv.org/abs/1911.00172) | Motivates nearest-neighbor memory for sequence prediction when exact local contexts repeat. |
| Retrieval-augmented generation | [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) | General pattern: combine parametric model with explicit retrieval. |
| Constrained decoding | [Scholak et al., PICARD](https://arxiv.org/abs/2109.05093) | Strong analogy for rejecting invalid generations with an incremental constraint checker. |
| Constrained beam search | [Hokamp and Liu, Lexically Constrained Decoding](https://aclanthology.org/P17-1141/) | Shows decoding can be guided by hard constraints rather than model probabilities alone. |
| Exposure bias | [Bengio et al., Scheduled Sampling](https://arxiv.org/abs/1506.03099) | Explains why greedy Task 2 rollout can compound mistakes. |
| RL sequence training | [Ranzato et al., Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/abs/1511.06732) | RL-style sequence objectives can help after supervised baselines, but are not the first move here. |
| Self-critical RL | [Rennie et al., Self-Critical Sequence Training](https://arxiv.org/abs/1612.00563) | A mature sequence-level RL pattern, useful later if optimizing exact-match or edit-distance rewards. |

## Candidate Method Families

### 1. Pure n-gram / Markov statistics

What it does:

- Counts what usually follows recent steps.
- Fast, transparent, and reproducible.
- Good for local next-step prediction in structured routes.

Fit:

- Strong Task 1 baseline.
- Weak Task 2 because greedy rollout compounds errors.
- Does not prove deep process understanding.

Verdict:

- Keep as baseline and fallback.
- Not enough for final solution.

### 2. Retrieval / nearest historical continuation

What it does:

- Given a partial route, find training prefixes with similar recent context near
  the same progress fraction.
- Reuse the corresponding historical suffix as a completion.

Why it fits this track:

- Official valid eval cuts are exactly 60% and 80%.
- Generated process traces have repeated local process blocks.
- Completion wants a long suffix, not just one next token.
- Retrieval avoids the n-gram baseline's repeated-cycle failure mode.

Verdict:

- Best immediate upgrade for Solution 1.
- It is not a final learned model, but it is a very strong engineering move
  for the available data and time.

### 3. LSTM / GRU sequence models

What they do:

- Learn sequential dynamics from event logs.
- Well-established in predictive process monitoring.

Fit:

- Better than n-grams for longer dependencies.
- Easier to train than a transformer.
- Less compelling than a small transformer for the final "model competency"
  story, especially because the repo plan already points to a GPT-style model.

Verdict:

- Reasonable fallback if transformer training is unstable.
- Not selected for Solution 1 because retrieval is cheaper and immediately
  improves the largest baseline weakness.

### 4. Step-level transformer

What it does:

- Treats each process step as one token.
- Learns long-range dependencies with self-attention.

Fit:

- Strong candidate for the final winning model.
- Should be trained from random init on generated step sequences.
- Needs real training logs, loss curves, and leave-one-family-out evaluation.

Verdict:

- Best Solution 2 / hero-model direction.
- Not implemented as Solution 1 because it needs more time and compute to be
  credible.

### 5. Symbolic validator / process constraints

What it does:

- Checks whether a sequence violates explicit process rules.

Fit:

- Perfect for Task 3 when the public validator rules match the eval rules.
- Useful for constrained decoding and reranking in future solutions.
- But direct use is an oracle, not learned model behavior.

Verdict:

- Keep for Task 3 and for future constrained decoding.
- Be honest in reports and demos.

### 6. Grammar-constrained decoding

What it does:

- Generates candidate next steps or suffixes while rejecting invalid steps
  according to process constraints.

Fit:

- Very promising for Task 2 exactness and Task 3 validity.
- Harder to implement correctly because the public validator checks complete
  or partial traces after the fact, not an explicit next-token automaton.

Verdict:

- Strong future improvement.
- Solution 1 uses retrieval first and keeps validator-gating as the conceptual
  next step.

### 7. Reinforcement learning

What it does:

- Optimizes a policy against a reward.
- For sequence generation, rewards can be exact match, edit distance, validity,
  or rule attribution.

Fit:

- There is no interactive fab environment in this repo.
- The reward is offline and sparse.
- RL would add instability before we have a strong supervised transformer.

Verdict:

- Not the right first solution.
- Consider later as sequence-level fine-tuning after a supervised transformer
  is trained and evaluated.

## Decision Matrix

Scores are qualitative: 1 = weak fit, 5 = strong fit.

| Method | Task 1 next step | Task 2 completion | Task 3 anomaly | Task 4 OOD | Implementable now | Jury story | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N-gram only | 4 | 1 | 2 | 3 | 5 | 2 | 17 |
| Retrieval + n-gram | 4 | 4 | 2 | 3 | 5 | 3 | 21 |
| Validator only | 1 | 1 | 5 | 2 | 5 | 2 | 16 |
| LSTM/GRU | 4 | 3 | 3 | 3 | 3 | 3 | 19 |
| Step-level transformer | 5 | 4 | 4 | 4 | 2 | 5 | 24 |
| Transformer + constraints | 5 | 5 | 5 | 4 | 2 | 5 | 26 |
| RL-first | 3 | 3 | 3 | 2 | 1 | 3 | 15 |

## Recommendation

Build in stages:

1. **Solution 1 now:** retrieval + n-gram + validator hybrid.
2. **Solution 2 next:** step-level transformer trained from scratch.
3. **Solution 3 stretch:** transformer with validator-guided reranking or
   constrained decoding.
4. **RL only after that:** sequence-level fine-tuning if exact-match/edit
   distance remains the bottleneck.

## What Solution 1 Implements

`solution_1_hybrid_retrieval` implements the immediate recommendation:

- Build a context index over training prefixes near the 60% and 80% official
  cut points.
- For Task 1, combine retrieval votes with trigram top-k votes.
- For Task 2, choose a retrieved historical suffix instead of greedy n-gram
  rollout.
- For Task 3, use the public validator oracle.
- For Task 4 proxy, evaluate leave-one-family-out next-step performance.

## Local Result Summary

Compared with Solution 0:

| Metric | Solution 0 | Solution 1 |
| --- | ---: | ---: |
| Task 1 Top-1 | 0.6800 | 0.7283 |
| Task 1 Top-3 | 0.9883 | 1.0000 |
| Task 1 Top-5 | 1.0000 | 1.0000 |
| Task 1 MRR | 0.8336 | 0.8633 |
| Task 2 exact match | 0.0000 | 0.0017 |
| Task 2 normalized edit distance | 0.6152 | 0.2420 |
| Task 2 token accuracy | 0.2073 | 0.4485 |
| Task 2 block accuracy | 0.3922 | 0.7167 |
| Task 3 accuracy/F1/AUC/rule attribution | 1.0000 | 1.0000 |

Main takeaway:

- Retrieval directly addresses the baseline's biggest weakness: long suffix
  completion.
- Task 3 is still oracle-perfect, so it should not be presented as learned
  anomaly intelligence.
