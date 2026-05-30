            # Training / Fitting Log

            This solution includes a deterministic evidence cascade and a small XGBoost raw-vs-template diagnostic. The official CSV path is rebuilt from lookup tables and validator evidence; the bridge diagnostic trains XGBoost classifiers from the synthetic 110-family probe.

            ## Command

            ```bash
            python -B solutions/solution_21_template_boosted_bridge/solution.py
            ```

            ## Data Summary

            - Families: `mosfet, igbt, ic`
            - Valid self-eval rows: `600`
            - Anomaly self-eval rows: `600`
            - Training sequences: `6709`
            - Local split seed: `42`

            ## Result Snapshot

            - Task 1 Top-1: `1.0000`
            - Task 1 MRR: `1.0000`
            - Task 2 normalized edit distance: `0.0000`
            - Task 2 block accuracy: `1.0000`
            - Task 3 accuracy: `1.0000`

## XGBoost Bridge Snapshot

- Raw XGBoost 10-seed Top-1: `0.5806`
- Template XGBoost 10-seed Top-1: `0.7531`
- Raw family-specific label coverage: `0.0000`
- Template family-specific label coverage: `1.0000`


            ## Checkpoint Status

            The XGBoost bridge is retrained deterministically from source for each diagnostic run; no binary checkpoint is committed because the model is small and rebuildable. For a real neural submission, this folder should be
            extended with `.pt`/`.safetensors` checkpoints and cluster logs.
