# Slide outline (≤10) — Learning & Benchmarking Process Logic

> Speaker deck for the Industrial AI / Infineon track. Lead with KREMSIANS (honest OOD)
> and the model-size-vs-OOD finding. Each slide = one idea. Numbers are self-eval estimates;
> the submitted CSVs (`submissions_official/`) are organiser-scored.

---

### 1. Title
**Does the model learn process logic — or just memorise?**
From-scratch GPT on semiconductor process sequences · trained on CINECA Leonardo (A100).
Team `zero_one_hack_0625`. Open stack, own training on the cluster — no API, no lookup.

### 2. The problem in one picture
- 115–150-step fab sequences; 3 known families (MOSFET, IGBT, IC); shared grammar + family blocks.
- 3 scored tasks (next-step, completion, anomaly) + a **hidden 4th family** (Task 4 = the real test).
- The trap: retrieval/lookup solutions ace Tasks 1–3 and collapse on Task 4.

### 3. Our approach (the stack)
- **Step-level tokeniser**: 1 process step = 1 token (~198-token closed vocab) → optimises exactly what's scored.
- **From-scratch GPT** at 3 sizes (0.6M / 4.9M / 21.5M) + n-gram baseline + Qwen2.5 ladder.
- **Grammar-guided decoding**: mask logits to grammar-valid successors (~1.9 choices/position).
- **Symbolic validator** for anomaly: provably correct, family-agnostic.

### 4. Baseline → trained (in-distribution, Task 1)
| Model | Top-1 | Top-5 | MRR |
|---|---|---|---|
| n-gram | 0.73 | 0.99 | 0.85 |
| GPT:large (21.5M) | **0.81** | **1.00** | **0.90** |

Top-1 ~0.81 ≈ the grammar's irreducible ceiling (synonyms, optional steps). We don't chase noise.

### 5. **KREMSIANS** — our honest 4th family (the centrepiece)
- Self-built, grammatically valid, structurally **novel** (hybrid prep, 5 litho cycles, dual implant).
- Designed *after* freezing all model choices; uses only the existing vocab (fair OOD).
- Never trained on → a clean stand-in for the hidden Task 4.

### 6. KREMSIANS result (the money slide)
| GPT:large on KREMSIANS | no guide | + guide |
|---|---|---|
| Top-1 | 0.59 | 0.58 |
| Top-3 | 0.80 | **0.87** |
| Top-5 | 0.83 | **0.97** |

**Correct next step in top-5 ~97% of the time on a family it has never seen.** That's transferable logic.

### 7. Scaling finding: bigger ≠ more transferable
| Model | Params | ID Top-1 | KREMSIANS Top-1 | KREMSIANS Top-5 +guide |
|---|---|---|---|---|
| GPT:tiny | 0.6M | 0.81 | 0.587 | 0.967 |
| GPT:large | 21.5M | 0.81 | 0.591 | 0.969 |

35× params → flat on every axis. Capacity isn't the bottleneck; the family barrier is.

### 8. Anomaly detection (Task 3) — solved, and it generalises
- Symbolic `validate_sequence()` → **F1 = 1.00**, perfect rule attribution.
- 10 rules are **family-agnostic** → carries straight over to the hidden 4th family.
- Model surprisal supplies the continuous ROC-AUC `SCORE`.

### 9. Reproducibility & sovereignty
- One command on Leonardo: `sbatch job.slurm` → trains ladder → official-format CSVs → self-eval scores → KREMSIANS OOD.
- Official IDs (`valid_0001`/`anomaly_xxxx`) scored by the organiser's own `eval_metrics.py`.
- Entirely open stack (PyTorch + HF), trained on the EU cluster — model competency, not a wrapper.

### 10. Demo + takeaway
- **Side-by-side demo**: n-gram baseline vs trained GPT (± grammar guide) on the same partial sequence.
- Takeaway: a small, own-trained model + grammar structure beats memorisation — and proves it on an unseen family.

---

## Demo script (≤2 min)
1. Pick one KREMSIANS partial sequence. Show n-gram top-5 vs GPT top-5 vs GPT+guide top-5 → guide nails the correct step.
2. Show an injected violation → `validate_sequence()` flags it and names the rule; surprisal spikes on the offending step.
3. Cut to the scaling table: "bigger memorised harder, didn't generalise." End on the 97% top-5 OOD line.
