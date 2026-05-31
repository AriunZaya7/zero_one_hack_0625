# LLM Sequence Ruling Attempts

This folder collects attempts to derive sequence-ordering knowledge from step descriptions, parameter hints, public semiconductor-process references, and the generated traces.

These artifacts are intended as reusable inputs for future solutions. They are not official ground truth and should not be presented as real Infineon production routing.

## Attempts

| Attempt | Focus | Main output | Best use |
|---|---|---|---|
| `attempt_0` | LLM + public research candidate rules | `candidate_rules.json` | Rule priors, explanations, constrained decoding |
| `attempt_1` | Empirical support from generated traces | `empirical_rule_support.json` | Decide which rules can be hard constraints vs soft penalties |
| `attempt_2` | Description-aware feature/context pack | `description_context_pack.json` | Step enrichment, duplicate-step disambiguation, reranking features |

## Practical Use In A Future Solution

1. Use `attempt_2` to enrich every candidate step with semantic context.
2. Use `attempt_1` to choose penalties and confidence weights.
3. Use `attempt_0` to explain why a candidate is plausible or invalid.
4. Combine with n-gram/retrieval/transformer scores rather than replacing them.

Suggested scoring sketch:

```text
score(candidate_next) =
    model_score
  + retrieval_score
  + semantic_block_bonus
  - empirical_rule_penalty
  - hard_validator_penalty
```

