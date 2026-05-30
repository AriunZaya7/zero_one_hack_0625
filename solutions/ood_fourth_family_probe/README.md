# Hypothetical Hidden-Family OOD Probe

This folder stress-tests the final Industrial AI strategy against a locally
invented fourth and fifth family named exactly:

- `THEFOURTHFAMILY`
- `THEFIFTHFAMILY`

These families are meant to be plausible, not official. They simulate what
organizers could do for hidden OOD:

- keep the same public process-grammar style,
- add new exact strings,
- add family-specific power-device blocks,
- still use the public validation philosophy so the task is fair rather than
  random.

The two families intentionally share the same broad generator architecture. The
point of the probe is not to invent two unrelated tasks; it is to test whether
the final strategy behaves correctly when exact hidden-family names and strings
change.

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
- `decoupled_known_fallback_only`: no matching hidden-family full routes are
  visible, and the fallback only knows MOSFET/IGBT/IC.
- `decoupled_visible_family_routes`: related hidden-family full routes are
  visible but are not the exact target routes.
- `oracle_hidden_family_generator_training`: the hidden-family generator spec is
  known and can create training data, but exact target routes are still hidden.

This probe is intentionally separate from official scoring. It exists to make
our OOD claims more concrete and to show where the final strategy is strong or
information-limited.
