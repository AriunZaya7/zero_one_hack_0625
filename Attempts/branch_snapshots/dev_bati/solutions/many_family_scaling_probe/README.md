# Many-Family Scaling Probe

This folder tests a specific OOD question:

> If we invent 110 plausible process families, hold out the same 10 validation
> families, and increase the number of training families from 2 to 100, which
> metrics keep improving and which metrics hit an information limit?

The probe is local and synthetic. It is not an official benchmark. It exists to
make the final pitch more concrete.

The setup is:

- 110 total synthetic families named `SYNTHFAMILY001` through
  `SYNTHFAMILY110`.
- Training pool: `SYNTHFAMILY001` through `SYNTHFAMILY100`.
- Fixed validation pool: `SYNTHFAMILY101` through `SYNTHFAMILY110`.
- Train-family counts: 2, 5, 10, 20, 40, 60, 80, and 100.
- Every train count is evaluated on the same 10 validation families.

Run from the repository root:

```bash
python -B solutions/many_family_scaling_probe/family_scaling_probe.py
```

The script writes:

- `outputs/metrics_by_count.csv`
- `outputs/metrics_by_count.json`
- `outputs/per_family_metrics.csv`
- `outputs/per_family_metrics.json`
- `outputs/README.md`
- `analysis.html`

The probe reports two model variants:

- `raw_exact_retrieval`: uses exact strings from the training families only.
- `template_adapted_retrieval`: normalizes steps like
  `SYNTHFAMILY003 JTE DOSE VERIFICATION` into a family placeholder during
  training, then rewrites the placeholder back to the visible validation family
  name at prediction time.

This is the key distinction:

- Raw exact retrieval tests what happens if the new family's exact strings are
  not inferable.
- Template-adapted retrieval tests whether the exact strings can be derived from
  the visible family name plus learned process-step suffixes.
