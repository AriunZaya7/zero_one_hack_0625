# Comparison: Ground Truth vs XGBoost IC+IGBT -> MOSFET OOD

---

Here's the analysis comparing the learned rules against ground truth:

### 1. Summary Table

| Rule | Ground Truth | Model Learned | Similarity (0-1) | Match? |
|------|--------------|---------------|------------------|--------|
| BACKSIDE_METAL_AFTER_PASSIVATION | Yes (implied by passivation sequence) | Weak signal (1.08x) | 0.2 | Partial |
| RULE_DEP_NO_CLEAN | Explicit rule | Not learned | 0.0 | No |
| RULE_METAL_ETCH_NO_LITHO | Explicit rule | Not learned | 0.0 | No |
| RULE_ETCH_NO_MASK | Explicit rule | Inverse signal | -0.1 | No |
| RULE_LITHO_LEVEL_SKIP | Explicit rule | Not learned | 0.0 | No |
| RULE_IMPLANT_NO_MASK | Explicit rule | Not learned | 0.0 | No |
| RULE_CMP_NO_DEP | Explicit rule | Inverse signal | -0.1 | No |
| RULE_PAD_OPEN_BEFORE_DEP | Explicit rule | Inverse signal | -0.1 | No |
| RULE_TEST_BEFORE_PASSIVATION | Explicit rule | Not learned | 0.0 | No |
| RULE_SHIP_BEFORE_TEST | Explicit rule | Not learned | 0.0 | No |

### 2. Per-Rule Gap Analysis

1. **BACKSIDE_METAL_AFTER_PASSIVATION**:
   - Ground Truth: Implied by passivation sequence requirements
   - Learned: Very weak signal (1.08x ratio)
   - Gap: Model barely detects this global constraint

2. **All Other Rules**:
   - Ground Truth: Explicit process constraints
   - Learned: Either no detection or inverse signals
   - Gap: Complete failure to learn fundamental process constraints

### 3. Overall Similarity Score
- **Weighted Average Similarity: 0.01/1.0**
- Breakdown:
  - 1 partial match at 0.2 similarity
  - 6 complete misses (0.0)
  - 3 anti-correlations (-0.1)

### 4. Final Verdict: Patterns or Genuine Logic?
**Conclusion:** The model shows no meaningful understanding of semiconductor process logic. 

Key evidence:
1. **Inverse Signals**: For 3 rules, the model actually assigns lower perplexity to invalid sequences (0.82x-0.86x ratios)
2. **Missed Fundamentals**: Failed to learn any of the 10 explicit rules
3. **Weak Single Signal**: The one detected "rule" shows minimal effect size (1.08x) and may be statistical noise

**Diagnosis:** The model is performing surface-level pattern matching from the IC+IGBT training data, with zero transferable understanding of process constraints. The results suggest the model cannot generalize to MOSFET process validation, nor does it understand even basic semiconductor manufacturing principles like clean-before-deposition or mask-before-etch.

**Recommendation:** This architecture appears fundamentally unsuitable for OOD process validation. Either:
1. Train on explicit rule violations, or
2. Use architectures with stronger symbolic reasoning capabilities, or
3. Implement explicit rule-checking post-processing