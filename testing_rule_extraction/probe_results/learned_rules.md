# Learned Rules — XGBoost IC+IGBT -> MOSFET OOD Behavioral Probing

> **Model:** XGBoost IC+IGBT -> MOSFET OOD
> **Method:** Perplexity probing on valid vs rule-violated sequences
> **Note:** LLM never saw generation_rules.md — inferred from behavior only

---

Here's the analysis of the probe results:

## Learned Rule: BACKSIDE_METAL_AFTER_PASSIVATION
**Observed behavior:** Model assigns 1.08x higher perplexity when backside metal is deposited before passivation
**Inferred constraint:** DEPOSIT BACKSIDE METAL must be preceded by DEPOSIT PASSIVATION and CURE PASSIVATION
**Confidence:** LOW
**Evidence:** valid_ppl=1.7355, invalid_ppl=1.878, ratio=1.0821x
**Rule type:** GLOBAL

## Overall Assessment
The model shows extremely weak learning of manufacturing rules, with only one rule showing even marginal detection (and that at very low confidence). The model appears to be largely pattern-matching without deep understanding of semiconductor process flows.

## Rules Learned (list)
1. BACKSIDE_METAL_AFTER_PASSIVATION (weak signal)

## Rules NOT Learned (list with reason)
1. RULE_DEP_NO_CLEAN (ratio ~1.0)
2. RULE_METAL_ETCH_NO_LITHO (ratio ~1.0)
3. RULE_ETCH_NO_MASK (ratio 0.82x - inverse of expected)
4. RULE_LITHO_LEVEL_SKIP (ratio 1.0)
5. RULE_IMPLANT_NO_MASK (ratio ~1.0)
6. RULE_CMP_NO_DEP (ratio 0.86x - inverse of expected)
7. RULE_PAD_OPEN_BEFORE_DEP (ratio 0.83x - inverse of expected)
8. RULE_TEST_BEFORE_PASSIVATION (ratio 0.96x)
9. RULE_SHIP_BEFORE_TEST (ratio 0.98x)

## Key Insight: Local vs Global Rules
The model shows no ability to learn either local or global process constraints. The one weak signal we see is for a global rule, but the effect size is minimal.

## What This Tells Us (patterns vs genuine logic?)
The model is likely learning surface-level sequence patterns rather than genuine process constraints. The perplexity ratios suggest it cannot reliably distinguish valid from invalid manufacturing sequences. This is particularly concerning given that most violations show either no difference or sometimes even lower perplexity for invalid sequences (inverse of expected behavior). The model appears poorly suited for out-of-distribution MOSFET process validation when trained only on IC+IGBT data.