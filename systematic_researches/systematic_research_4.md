# Systematic Research 4: Judge-Aware Portfolio Selection

Date: 2026-05-30

Target solution: `solutions/solution_9_judge_aware_portfolio`

## Research Question

The current solution suite has specialists that are best at different metrics.
The question for this pass was:

> Should the final visible-task submission be a single base model, or a
> transparent portfolio that selects the strongest completed specialist per
> judging task?

## External Research Direction

This pass looked for work around ensembles, model selection, and portfolio
methods in predictive process monitoring and event-log prediction. I used
primary paper pages and abstracts, not blog posts, for the decision framing.

Relevant papers:

- Tax, Verenich, La Rosa, and Dumas, "Predictive Business Process Monitoring
  with LSTM Neural Networks" (CAiSE 2017):
  https://arxiv.org/abs/1612.02130
  This paper is important because it treats process monitoring as a family of
  related but not identical tasks. It discusses next-event prediction and then
  uses next-task models to predict full continuations. That supports the idea
  that Task 1 and Task 2 can be related while still having different best
  implementations.
- Bukhsh, Saeed, and Dijkman, "ProcessTransformer: Predictive Business Process
  Monitoring with Transformer Network" (2021):
  https://arxiv.org/abs/2104.00721
  This paper is important for the future neural direction: transformers can
  model longer process traces than simple local models. For this pass, however,
  we did not start a half-finished transformer because the heartbeat requirement
  was to commit only runnable and measurable solutions.
- Tama, Comuzzi, and Ko, "An empirical investigation of different classifiers,
  encoding and ensemble schemes for next event prediction using business
  process event logs" (2020): https://arxiv.org/abs/2008.10748
  This is the most directly relevant paper for Solution 9. It evaluates
  next-event prediction with multiple classifiers, encodings, and ensemble
  schemes, and it frames model choice as data- and task-dependent. That matches
  what we see locally: no one completed solution dominates every metric.
- Weinzierl, Dunzer, Zilker, and Matzner, "Prescriptive Business Process
  Monitoring for Recommending Next Best Actions" (2020):
  https://arxiv.org/abs/2008.08693
  This is the useful reference for the earlier online-RL question. It focuses
  on next-best-action recommendations and simulation/KPI optimization. That is
  closer to reinforcement-learning-style control than our current offline CSV
  prediction task.
- Rice, "The Algorithm Selection Problem" (1976; Purdue technical report page):
  https://docs.lib.purdue.edu/cstech/99/
  This is the classic algorithm-selection framing: choose among algorithms based
  on which one performs best for the problem instance or performance measure.
  Solution 9 is a small, explicit version of that idea: choose one specialist
  for each submitted judging file.

The useful lesson is general rather than tied to one exact architecture:
event-log tasks often reward different properties depending on whether the
target is next activity, remaining suffix, anomaly classification, or
out-of-distribution generalization. Heterogeneous ensembles and
algorithm-selection layers are therefore reasonable when the component models
have complementary strengths.

Relevant research themes:

- Predictive process monitoring has separate subproblems: next-activity
  prediction, suffix prediction, outcome prediction, and anomaly detection.
- Ensembles are commonly used when no single learner dominates every metric.
- Algorithm selection is most defensible when the selection rule is explicit,
  measured, and does not hide tradeoffs.
- Online RL is better suited to settings with actions, feedback, and a reward
  signal. The current submission is offline prediction: the judge gives hidden
  sequences, we emit CSV predictions, and there is no interactive reward loop.

## Local Evidence Used

The regenerated 10-seed report showed:

- Solution 3 has the best mean Task 1 Top-1 among the completed deterministic
  attempts: `0.6970`.
- Solution 7 and Solution 8 have the best mean Task 2 normalized edit distance:
  `0.2334`.
- Solution 8 improves the Task 3 explanation path by avoiding direct
  `validate_sequence()` inference.

The seed-42 visible metrics showed:

- Solution 3 Task 1 Top-1: `0.7350`
- Solution 7/8 Task 2 normalized edit distance: `0.2333`
- Solution 8 Task 3 accuracy and rule attribution: `1.0000`

## Chosen Design

Solution 9 selects specialists by task:

- Task 1: Solution 3 synthetic-augmented retrieval
- Task 2: Solution 7 Monte Carlo suffix library
- Task 3: Solution 8 semantic conformance checker

This is not a new base learner. It is a transparent judge-aware model-selection
layer.

## Why Not Online RL For This Committed Attempt?

Online RL is worth discussing for the product story, but it is not the best
committed method for this offline hackathon artifact.

For RL to be appropriate, we would need:

1. A state: current partial process route.
2. Actions: allowed next manufacturing steps.
3. A transition model or simulator: what happens after the chosen action.
4. A reward: yield, cycle time, validity, cost, or official score feedback.
5. An exploration policy: how the model tries actions without harming a real
   manufacturing process.

The public track materials give us historical/generated sequences and a
validator, but not an interactive hidden-eval environment. The official judging
surface is therefore closer to supervised sequence prediction than online RL.

The practical compromise is:

- Now: use deterministic retrieval, public-generator augmentation, conformance
  logic, and a transparent portfolio that produces measurable CSVs.
- Later: if the team builds a simulator from the public generator/validator,
  train a policy offline or in simulation and compare it against the portfolio.
  That would belong in a later `solution_10` only if we can make it runnable,
  measurable, and honest.

## Why This Is Legitimate

The official output files are task-separated:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`

Because each task is submitted as a separate output artifact, using the best
available specialist per task is a practical and auditable strategy. The
portfolio does not secretly mix hidden labels; it only combines already
completed runnable methods.

## Expected Strengths

- Best visible seed-42 Task 1 Top-1/MRR among completed attempts.
- Best visible seed-42 Task 2 edit distance and block accuracy among completed
  deterministic attempts.
- Best current explainable Task 3 path.
- Clear presentation story: the team measured tradeoffs and selected the right
  specialist for each scoring surface.

## Expected Weaknesses

- Hidden-family generalization is weaker than the Solution 2/Solution 8 family
  on IC leave-one-family-out because Task 1 uses Solution 3.
- It is a portfolio, not a learned unified process model.
- It does not replace the need for a real Leonardo-trained transformer if time
  allows.

## Submission Recommendation

Use Solution 9 if the team wants the strongest known visible-task CSV bundle.
Use Solution 8 if the team wants a more conservative hidden-family story while
keeping the stronger completion and semantic anomaly path.
