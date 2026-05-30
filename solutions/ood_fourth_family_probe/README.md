# Hypothetical Fourth-Family OOD Probe

This folder stress-tests the final Industrial AI strategy against a locally
invented fourth family named `sic_power`.

The family is meant to be plausible, not official. It simulates what organizers
could do for hidden OOD:

- keep the same public process-grammar style,
- add new exact strings,
- add family-specific SiC/power-device blocks,
- still use the public validation philosophy so the task is fair rather than
  random.

Run from the repository root:

```bash
python -B solutions/ood_fourth_family_probe/fourth_family_probe.py
```

The script writes:

- `outputs/per_seed_metrics.csv`
- `outputs/per_seed_metrics.json`
- `outputs/summary_metrics.csv`
- `outputs/summary_metrics.json`
- `outputs/README.md`

The important scenarios are:

- `coupled_exact_route_memory`: full valid routes are visible and match the
  partial rows, like the released official files.
- `decoupled_known_fallback_only`: no matching fourth-family full routes are
  visible, and the fallback only knows MOSFET/IGBT/IC.
- `decoupled_visible_family_routes`: related fourth-family full routes are
  visible but are not the exact target routes.
- `oracle_fourth_family_generator_training`: the fourth-family generator spec is
  known and can create training data, but exact target routes are still hidden.

This probe is intentionally separate from official scoring. It exists to make
our OOD claims more concrete and to show where the final strategy is strong or
information-limited.
