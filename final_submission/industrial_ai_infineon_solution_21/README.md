# Final Industrial AI Submission

This folder is the final hackathon submission bundle for the Industrial AI
track.

## Upload These Three Scored Files

Only these files are intended as scored prediction CSVs:

```text
scored_csvs/nextstep.csv
scored_csvs/completion.csv
scored_csvs/anomaly.csv
```

They match `training_data/generation_rules.md` Section 5.3:

| File | Required header | Required rows |
| --- | --- | ---: |
| `nextstep.csv` | `EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5` | `600` |
| `completion.csv` | `EXAMPLE_ID,PREDICTED_SEQUENCE` | `600` |
| `anomaly.csv` | `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE` | `987` |

## Supporting Attachments

The `attachments/` folder is supporting evidence for the report, slides, or demo.
Do not upload those files as scorer inputs unless the form explicitly asks for
supporting material.

Important attachments:

- `FINAL_PITCH_OOD_STRATEGY.html`
- `solution_21_explanation.html`
- `solution_21_metrics.md`
- `solution_21_metrics.json`
- `template_boosting_10_seed.csv`
- `template_boosting_curve.csv`
- `template_boosting_examples.csv`
- `template_boosting_tree_shap.csv`
- `template_boosting_xgb_training_logloss.csv`
- `template_boosting_bridge.json`
- `raw_xgb_exact_seed0_xgboost_model.json`
- `template_xgb_bridge_seed0_xgboost_model.json`
- `template_xgb_feature_names.json`
- `solution_21_training_log.md`
- `solution_21_checkpoint_manifest.md`
- `solution_21_loss_curve.csv`
- `baseline_vs_model_examples.md`
- `pitch_slides_outline.md`
- `video_script.md`
- `paired_length_guard_audit.csv`
- `submission_readiness_audit.md`

## Manual Legal Checklist

Before the Tally submission, the team must still confirm the external items
that cannot be changed from this folder:

- Repository is public.
- Repository root has the MIT `LICENSE`, `README.md`, `REPORT.md`, and
  `requirements.txt`.
- Slides are exported as PDF.
- Demo video is under two minutes.
- No file from `attachments/` is uploaded as one of the three scorer CSVs.

## Verify Before Submit

Run from the repository root:

```bash
python -B final_submission/industrial_ai_infineon_solution_21/VERIFY_FINAL_SUBMISSION.py
```

Expected result:

```text
FINAL SUBMISSION OK
```

## Regenerate Source Outputs

Run from the repository root:

```bash
python -B solutions/solution_21_template_boosted_bridge/solution.py
python -B solutions/prepare_submission_packages.py
```

Then copy the regenerated official CSVs into `scored_csvs/` if needed.
