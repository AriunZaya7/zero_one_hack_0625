# Zero One Hack_01

**36 hours. Real infrastructure. European AI sovereignty.**

Welcome to the central repository for Zero One Hack_01, hosted by [Lumos Consulting](https://lumos-consulting.at) at [AI Factory Austria](https://aifactory.at) in Vienna, with compute provided by CINECA on the Leonardo GPU Cluster.

---

## Quick links

- 🌐 **Docs**: [docs.zero-one.lumos-consulting.at](https://docs.zero-one.lumos-consulting.at/)
- 💬 **Discord**: https://discord.gg/e6rrVbcD5
- 📍 **Venue**: AI Factory Austria (AI:AT), Vienna

---

## Industrial AI Work In This Branch

This branch contains our Industrial AI (Infineon) solution attempts under
[`solutions/`](./solutions/). The current suite is documented in:

- [`OVERALL_SUMMARY.html`](./OVERALL_SUMMARY.html) - standalone end-to-end
  explanation of what we learned, including the generator/validator upper-bound
  result
- [`FINAL_PITCH_OOD_STRATEGY.html`](./FINAL_PITCH_OOD_STRATEGY.html) - final
  pitch for the OOD-ready strategy around new exact strings and new
  family-specific blocks
- [`solutions/ood_fourth_family_probe/`](./solutions/ood_fourth_family_probe/) -
  hypothetical fourth/fifth hidden-family OOD stress test used by the final
  pitch
- [`solutions/many_family_scaling_probe/analysis.html`](./solutions/many_family_scaling_probe/analysis.html) -
  110-family scaling probe with fixed 10-family validation curves
- [`solutions/solution_18_family_template_grammar/`](./solutions/solution_18_family_template_grammar/) -
  current OOD-facing candidate: exact route memory plus normalized
  `__FAMILY__` template fallback for derivable hidden-family strings
- [`REPORT.md`](./REPORT.md) - root submission report
- [`solutions/solutions_comparison.md`](./solutions/solutions_comparison.md) -
  metrics and solution comparison
- [`solutions/submission_readiness_audit.md`](./solutions/submission_readiness_audit.md) -
  checklist against the official submission requirements

Each solution has:

- a runnable `solution.py`
- `outputs/nextstep.csv`, `outputs/completion.csv`, `outputs/anomaly.csv`
- when official participant inputs are present,
  `outputs/official_submission/nextstep.csv`,
  `outputs/official_submission/completion.csv`, and
  `outputs/official_submission/anomaly.csv`
- local metrics in `outputs/metrics.md` and `outputs/metrics.json`
- a standalone `explanation.html`
- a manual `how_to_submit_this_solution.html`
- a complete `submission_package/` with `extras/results/`,
  `extras/training_artifacts/`, demo material, and a solution-level `REPORT.md`

`solutions/prepare_submission_packages.py` validates the exact Industrial AI
`training_data/generation_rules.md` §5 CSV headers before writing the readiness
audit, so format drift fails fast instead of silently reaching submission.

### Setup

Use Python 3.10 or newer. The current deterministic solution scripts use only
the Python standard library.

```bash
python -m pip install -r requirements.txt
```

### Run All Current Solution Attempts

From the repository root:

```bash
python -B solutions/solution_0_rule_mock/solution.py
python -B solutions/solution_1_hybrid_retrieval/solution.py
python -B solutions/solution_2_eval_aware_retrieval/solution.py
python -B solutions/solution_3_synthetic_augmented_retrieval/solution.py
python -B solutions/solution_4_length_aware_completion/solution.py
python -B solutions/solution_5_tuned_rank_ensemble/solution.py
python -B solutions/solution_6_alias_calibrated_retrieval/solution.py
python -B solutions/solution_7_monte_carlo_suffix_ensemble/solution.py
python -B solutions/solution_8_semantic_conformance_ensemble/solution.py
python -B solutions/solution_9_judge_aware_portfolio/solution.py
python -B solutions/solution_10_confidence_gated_consensus/solution.py
python -B solutions/solution_11_ood_guarded_consensus/solution.py
python -B solutions/solution_12_mbr_completion/solution.py
python -B solutions/solution_13_transductive_generator_validator/solution.py
python -B solutions/solution_14_synthetic_ml_generator_ensemble/solution.py
python -B solutions/solution_15_route_memory_mbr/solution.py
python -B solutions/solution_16_pseudolabel_metric_audit/solution.py
python -B solutions/solution_17_conformal_route_guard/solution.py
python -B solutions/solution_18_family_template_grammar/solution.py
python -B solutions/generate_official_submissions.py
python -B solutions/prepare_submission_packages.py
```

The official participant input files and `eval_metrics.py` are now present under
`tracks/industrial-infineon/participant_files/`. The organizers still withhold
final ground truth labels, so committed scores are local self-eval or
official-input diagnostics unless explicitly stated otherwise.

---

## The three tracks

| Track                | Partner  | What you'll build                                                                                                               |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 🧾 **Insurance AI**   | UNIQA    | An AI-guided conversion flow that replaces a static form-based insurance calculator. Persona-based simulations on Leonardo.     |
| ⚙️ **Industrial AI**  | Infineon | Train and benchmark sequence models on semiconductor process flows. Does your model learn real process logic, or just memorize? |
| 📈 **Forecasting AI** | Sybilion | Build a decision agent on top of a probabilistic forecasting API. Live mid-run plot twist on Sunday.                            |

Each track's full briefing, data, and starter materials live in [`/tracks/`](./tracks/). 

---

## What's provided

- **Compute**: Leonardo GPU Cluster (A100s). 
- **Workspace**: Power, fast WiFi, monitors on request, breakout rooms for team calls.
- **Mentors**: Domain experts from each partner company, plus ML/infra mentors from Lumos and HPE.
- **API credits and tokens**: Track-specific, documented in each track's README.


---
## How submissions work

1. **Fill out the Tally submission form** by Sunday 10:00 — link will be shared in `#announcements`
2. The form takes four fields: team name, repository URL, slides (PDF), and demo video (file or link, max 2 minutes)
3. The Tally form timestamp is your official submission time
4. After 10:00 the form closes. No late submissions.

Full submission details, requirements, and the pre-submission checklist live in [`/submission/SUBMISSION.md`](./submission/SUBMISSION.md).

---

## Judging

Each track has its own rubric in [`/judging/rubrics.md`](./judging/rubrics.md). All tracks share these baseline expectations:

- **Working artifact** — not a slideware demo, something that actually runs
- **Reproducibility** — your repo should let someone else re-run your work
- **Honest evaluation** — show what worked, show what didn't, show what you measured
- **Visible reasoning** — explain *why* you made the technical choices you did

---

## Code of conduct & house rules

- Be kind. Be useful. Be honest about your work.
- AI Factory Austria is a working facility — respect equipment, doors, quiet hours.
- Mentors are here to unblock you, not to write your code. Use them well.
- The Leonardo cluster is shared infrastructure. No cryptomining, no training on copyrighted data, no abuse of compute. Violations = disqualification.
- See [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) for the full version.

---

## Get help

| Channel                                    | Use for                             |
| ------------------------------------------ | ----------------------------------- |
| `#announcements`                           | Schedule changes, important updates |
| `#industrial`,`#insurance`, `#forecasting` | Track-specific questions            |
| `#infra`                                   | Leonardo, GPU quota, WiFi, hardware |
| `#general`                                 | Everything else                     |
| In-person Lumos desk (lobby)               | Anything urgent                     |

---

*Looking forward to seeing what you build.* 🚀
