# Systematic Research 2: Task-Specialized Suffix Completion

Date: 2026-05-30

Target solution: `solutions/solution_7_monte_carlo_suffix_ensemble`

## Research Question

Earlier deterministic solutions had strong Task 1 Top-k coverage but weaker
Task 2 full-completion quality. The question for this research pass was:

> Should completion be solved by the same next-step model, or should it use a
> full-suffix candidate distribution or suffix-library retrieval method?

The answer from both literature and local experiments is that completion should
be treated as a full suffix problem. Repeated next-step prediction can drift.
Whole-suffix retrieval or whole-suffix generation has a better match to Task 2.

## External Research Reviewed

### SuTraN, encoder-decoder transformer for suffix prediction

Source: <https://colab.ws/articles/10.1109%2Ficpm63005.2024.10680671>

SuTraN is directly relevant because it focuses on suffix prediction rather than
only one-step-ahead prediction. Its abstract says existing methods often focus
on one-step prediction and iterative feedback loops, while SuTraN predicts a
whole suffix with a transformer-style encoder-decoder approach.

Relevance to our repo:

- The track's Task 2 is exactly a suffix prediction task.
- The repo sequences are short enough for sequence models, but the committed
  deterministic solutions do not yet train the transformer hero model.
- Solution 7 is a non-neural approximation of the same lesson: do not treat
  completion as only repeated next-step prediction.

### ProcessTransformer

Source: <https://arxiv.org/abs/2104.00721>

ProcessTransformer argues that self-attention can capture long-range
dependencies in event logs and reports strong next-activity performance. This
supports the larger plan to train a from-scratch process transformer. However,
training a robust transformer is a bigger implementation step than one
incremental solution attempt.

Relevance to Solution 7:

- Solution 7 is not a transformer.
- It preserves long-range route information in a simpler way: by retrieving a
  whole suffix from a route library rather than only scoring the next local
  token.

### Hierarchical and Switch Transformer process monitoring work

Sources:

- <https://www.mdpi.com/2079-9292/12/6/1273>
- <https://link.springer.com/article/10.1007/s10115-025-02587-z>

These works reinforce that process-monitoring sequences can be long enough that
plain local models miss dependencies. Hierarchical modeling and mixture-of-expert
ideas are used to handle longer or more varied process traces.

Relevance to Solution 7:

- The Monte Carlo suffix library is a cheap deterministic stand-in for a
  mixture of possible future route variants.
- Different generated routes act like different candidate experts for the
  suffix.

### Tree and subtrace approaches for suffix prediction

Source: <https://mcml.ai/publications/rfm%2B25a/>

The MCML summary describes hierarchical structuring of subtrace patterns for
activity suffix prediction. The key idea is that mined control-flow subtraces
can compete with deep models for remaining-trace forecasting while keeping lower
model complexity.

Relevance to Solution 7:

- Solution 7 is not a tree implementation, but it is closer to this family of
  methods than to a neural model.
- It mines candidate continuations from a large set of valid traces and chooses
  a compatible suffix using retrieval.

### Probabilistic suffix prediction

Source: <https://arxiv.org/abs/2505.21339>

The uncertainty-aware ED-LSTM paper motivates predicting a distribution of
possible suffixes rather than only one most-likely suffix. This is important in
our track because the public generator has optional steps and aliases. There is
not always one deterministic future after a prefix.

Relevance to Solution 7:

- A large generated suffix library is a simple empirical distribution over
  possible futures.
- Retrieval selects one suffix from that distribution.
- Exact full-suffix match can stay low even when the predicted route is
  structurally close, because multiple valid futures exist.

## Local Repo Evidence Used

Local files reviewed:

- `CLAUDE.md`
- `PLAN.md`
- `training_data/generation_rules.md`
- `training_data/generate_sequences.py`
- `solutions/solution_1_hybrid_retrieval/solution.py`
- `solutions/solution_2_eval_aware_retrieval/solution.py`
- `solutions/solution_3_synthetic_augmented_retrieval/solution.py`
- `solutions/solution_6_alias_calibrated_retrieval/solution.py`

Important local observations:

- `PLAN.md` already says the hidden family probably shares vocabulary but
  differs in optional blocks and cycle counts.
- `training_data/generate_sequences.py` exposes a public valid-sequence
  generator with deterministic seeds.
- Solution 3 proved generator augmentation helps completion.
- Experiments for Solution 7 showed that using too much generated data for
  exact next-step ranking can reduce exact Top-1 because alias choices split
  votes across equivalent strings.
- The same larger generated library helps full suffix completion because Task 2
  rewards structural closeness across many future steps.

## Experiments Run During This Pass

The local experiments used the seed-42 split from
`solutions/solution_0_rule_mock/solution.py::load_split()`.

Completion-only generated library sweep:

| Generated sequences per family | Seed base | Exact match | Normalized edit distance | Token accuracy | Block accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 13,000 | 0.0033 | 0.2399 | 0.4651 | 0.7335 |
| 5,000 | 12,000 | 0.0000 | 0.2342 | 0.4693 | 0.7359 |
| 5,000 | 15,000 | 0.0000 | 0.2401 | 0.4595 | 0.7289 |
| 6,000 | 16,000 | 0.0000 | 0.2372 | 0.4673 | 0.7342 |
| 7,000 | 17,000 | 0.0033 | 0.2362 | 0.4689 | 0.7344 |
| 10,000 | 18,000 | 0.0067 | 0.2333 | 0.4737 | 0.7367 |

The 10,000-per-family / seed-base-18,000 setting was chosen because it gave the
best local completion edit distance and token accuracy in this sweep while
remaining practical to run locally.

## Chosen Solution Design

Solution 7 uses:

- Task 1: `solution_2_eval_aware_retrieval`
- Task 2: `solution_1_hybrid_retrieval` trained on public train sequences plus
  10,000 generated valid sequences per family
- Task 3: public validator oracle

This is deliberately task-specialized. It does not try to force one model to be
best at every metric.

## Expected Strengths

- Best local Task 2 normalized edit distance among committed deterministic
  attempts at creation time.
- Keeps the strong Solution 2 Task 1 Top-k and leave-one-family-out behavior.
- Uses the public generator in a reproducible way.
- Does not commit large generated CSVs.
- Good story for the jury: full completion is suffix prediction, not repeated
  single-step prediction.

## Expected Weaknesses

- Still not the final transformer or online RL solution.
- Exact Task 1 Top-1 remains alias-limited.
- Exact Task 2 full-suffix match remains low because valid futures are
  inherently variable.
- Task 3 is still an oracle validator, not learned anomaly detection.

## Implementation Target

Create `solutions/solution_7_monte_carlo_suffix_ensemble` with:

- runnable `solution.py`
- output CSVs for all three tasks
- metrics and leave-one-family-out proxy
- standalone `README.md`, `explanation.html`, and
  `how_to_submit_this_solution.html`
- generated `submission_package/`
