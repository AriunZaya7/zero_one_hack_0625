# Attempt 1 - Empirical Rule Support

This attempt tests description-derived ordering rules against the generated training traces.
It is not online research; it is corpus evidence from the repo itself.

Sequences analyzed: `3000`

## Rule Support

| Rule | Triggers | Satisfied | Support | Notes |
|---|---:|---:|---:|---|
| `EMP_001_CLEAN_BEFORE_DEPOSITION_LIKE` | 32000 | 31884 | 0.9964 | Tests clean-like context before oxidation/epitaxy/deposition. |
| `EMP_002_DEVELOP_BEFORE_PATTERNED_ETCH` | 18000 | 17000 | 0.9444 | Patterned etches should have developed resist nearby. |
| `EMP_003_STRIP_AND_CLEAN_AFTER_ETCH` | 18000 | 16498 | 0.9166 | Checks whether etch is followed by strip and clean within 8 steps. |
| `EMP_004_IMPLANT_FOLLOWED_BY_ACTIVATION` | 9000 | 9000 | 1.0000 | Looks for drive-in diffusion, RTA, or anneal after implant. |
| `EMP_005_CMP_AFTER_DEPOSITION_OR_FILL` | 6000 | 6000 | 1.0000 | Checks CMP has material to planarize nearby. |
| `EMP_006_TEST_AFTER_PASSIVATION_CURE` | 14763 | 14763 | 1.0000 | Electrical tests should appear after passivation cure. |
| `EMP_007_WAFER_SORT_BEFORE_SHIP` | 3000 | 3000 | 1.0000 | Terminal order sanity check. |
| `EMP_008_MEASUREMENT_SUBJECT_ALIGNMENT` | 19081 | 19081 | 1.0000 | Soft heuristic for whether a measurement follows what it measures. |

## Interpretation

- High support means the heuristic matches this synthetic corpus well.
- Lower support does not automatically mean the rule is wrong; it may be too strict for optional or context-dependent steps.
- Use low-support rules as soft reranking features, not hard validators.

## Files

- `mine_empirical_rule_support.py` - reproducible miner.
- `empirical_rule_support.json` - machine-readable output.
- `empirical_rule_summary.md` - this summary.
