"""
Classical boosted-tree next-step models.

Both XGBoost and CatBoost are trained as multiclass classifiers over
step-level next-token prediction. They expose the same methods as the other
solution models: top_k_next(), complete_sequence(), and sequence_log_prob().
"""

from __future__ import annotations

import math
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass, fields
from typing import Iterable

import numpy as np

from solution.data.vocab import Vocab
from solution.eval import rules as R

BOS = "<BOS>"
EOS = "<EOS>"
UNK_FAMILY = "<UNK_FAMILY>"


OP_PREDICATES = [
    ("clean", R.is_clean),
    ("deposit", R.is_deposit),
    ("develop", R.is_develop),
    ("etch", R.is_etch),
    ("metal_etch", R.is_metal_etch),
    ("implant", R.is_implant),
    ("cmp", R.is_cmp),
    ("fill", R.is_fill),
    ("pad_window", R.is_pad_window),
    ("test", R.is_electrical_test),
    ("passivation", lambda step: R.is_passivation(step) or R.is_cure(step)),
    ("backside", lambda step: "BACKSIDE" in step.upper() or R.is_backside_metal(step)),
    ("oxidation", R.is_oxidation),
    ("litho", lambda step: R.align_level(step) is not None),
]

EXTRA_COUNT_PREDICATES = [
    ("litho", lambda step: R.align_level(step) is not None),
    ("develop", R.is_develop),
    ("oxidation", R.is_oxidation),
    ("fill", R.is_fill),
    ("metal_etch", R.is_metal_etch),
    ("pad_window", R.is_pad_window),
    ("wafer_sort", R.is_wafer_sort),
    ("ship", R.is_ship),
]

RECENCY_PREDICATES = [
    ("clean", R.is_clean),
    ("deposit", R.is_deposit),
    ("etch", R.is_etch),
    ("implant", R.is_implant),
    ("cmp", R.is_cmp),
    ("test", R.is_electrical_test),
    ("passivation", lambda step: R.is_passivation(step) or R.is_cure(step)),
    ("litho", lambda step: R.align_level(step) is not None),
]


def records_from_families(families: dict[str, list[list[str]]], selected: set[str] | None = None) -> list[dict]:
    out = []
    for family, seqs in families.items():
        if selected is not None and family not in selected:
            continue
        out.extend({"family": family, "steps": list(seq)} for seq in seqs)
    return out


@dataclass
class BoostingConfig:
    context_size: int = 12
    max_seq_len: int = 256
    seed: int = 42
    beam_width: int = 5
    beam_branching: int = 8
    beam_length_norm: float = 0.7
    beam_rule_penalty: float = 4.0
    feature_version: int = 2


class BoostingStepModel:
    model_type = "boosting"

    def __init__(self, config: BoostingConfig | None = None):
        self.config = config or BoostingConfig()
        self.vocab: Vocab | None = None
        self.family_to_id: dict[str, int] = {UNK_FAMILY: 0}
        self.bigram_to_id: dict[tuple[int, int], int] = {}
        self.trigram_to_id: dict[tuple[int, int, int], int] = {}
        self.label_steps: list[str] = []
        self.step_to_label: dict[str, int] = {}
        self.model_label_ids: list[int] = []
        self.estimator = None
        self.class_priors: dict[int, float] = {}
        self._predict_cache: dict[tuple[str | None, tuple[str, ...]], dict[str, float]] = {}

    @property
    def cat_feature_indices(self) -> list[int]:
        start_lags = 20
        return [0] + list(range(start_lags, start_lags + self.config.context_size)) + [
            start_lags + self.config.context_size,
            start_lags + self.config.context_size + 1,
        ]

    @property
    def feature_names(self) -> list[str]:
        names = [
            "family_id",
            "prefix_len",
            "position_norm",
            "max_litho_level",
            "clean_count",
            "deposit_count",
            "etch_count",
            "implant_count",
            "cmp_count",
            "test_count",
            "passivation_count",
            "backside_count",
            "seen_clean",
            "seen_deposit",
            "seen_etch",
            "seen_implant",
            "seen_cmp",
            "seen_test",
            "seen_passivation",
            "seen_backside",
        ]
        names += [f"lag_{i}" for i in range(1, self.config.context_size + 1)]
        names += ["last_bigram_id", "last_trigram_id"]
        if self._feature_version >= 2:
            names += [
                "unique_step_count",
                "unique_step_frac",
                "repeat_step_frac",
                "last_litho_level",
                "litho_level_gap",
            ]
            names += [f"{name}_count_v2" for name, _ in EXTRA_COUNT_PREDICATES]
            names += [f"seen_{name}_v2" for name, _ in EXTRA_COUNT_PREDICATES]
            names += [f"last_is_{name}" for name, _ in OP_PREDICATES]
            names += [f"prev_is_{name}" for name, _ in OP_PREDICATES]
            names += [f"since_last_{name}_norm" for name, _ in RECENCY_PREDICATES]
        return names

    @property
    def _feature_version(self) -> int:
        return int(getattr(self.config, "feature_version", 1))

    def fit(
        self,
        train_records: list[dict],
        val_records: list[dict] | None = None,
        max_train_examples: int = 250_000,
        max_val_examples: int = 50_000,
        iterations: int = 1200,
        device: str = "cpu",
        verbose: bool | int = True,
        sample_strategy: str = "uniform",
        family_dropout: float = 0.0,
        **params,
    ):
        self._build_metadata(train_records)
        x_train, y_train = self._build_dataset(
            train_records,
            max_examples=max_train_examples,
            sample_strategy=sample_strategy,
            family_dropout=family_dropout,
        )
        x_val = y_val = None
        if val_records:
            x_val, y_val = self._build_dataset(
                val_records,
                max_examples=max_val_examples,
                sample_strategy=sample_strategy,
            )
        x_train, y_train, x_val, y_val = self._remap_for_estimator(x_train, y_train, x_val, y_val)
        self._fit_estimator(x_train, y_train, x_val, y_val, iterations, device, verbose, **params)
        return self

    def _fit_estimator(self, x_train, y_train, x_val, y_val, iterations, device, verbose, **params):
        raise NotImplementedError

    def _build_metadata(self, records: list[dict]):
        sequences = [record["steps"] for record in records]
        self.vocab = Vocab(sequences)

        families = sorted({record.get("family") or UNK_FAMILY for record in records})
        self.family_to_id = {UNK_FAMILY: 0}
        for family in families:
            if family not in self.family_to_id:
                self.family_to_id[family] = len(self.family_to_id)

        steps = [step for step in self.vocab.itos[Vocab.OFFSET:]]
        self.label_steps = steps + [EOS]
        self.step_to_label = {step: i for i, step in enumerate(self.label_steps)}

        bigrams = set()
        trigrams = set()
        for record in records:
            ids = [self._step_id(step) for step in record["steps"]]
            padded = [Vocab.BOS, Vocab.BOS] + ids
            for i in range(1, len(padded)):
                bigrams.add((padded[i - 1], padded[i]))
            for i in range(2, len(padded)):
                trigrams.add((padded[i - 2], padded[i - 1], padded[i]))
        self.bigram_to_id = {pair: i + 1 for i, pair in enumerate(sorted(bigrams))}
        self.trigram_to_id = {tri: i + 1 for i, tri in enumerate(sorted(trigrams))}

    def _remap_for_estimator(self, x_train, y_train, x_val, y_val):
        self.model_label_ids = sorted(int(y) for y in set(y_train.tolist()))
        original_to_model = {label_id: i for i, label_id in enumerate(self.model_label_ids)}
        y_train_model = np.asarray([original_to_model[int(y)] for y in y_train], dtype=np.int64)

        if x_val is None or y_val is None:
            return x_train, y_train_model, x_val, y_val

        keep = np.asarray([int(y) in original_to_model for y in y_val], dtype=bool)
        if not keep.any():
            return x_train, y_train_model, None, None
        x_val_model = x_val[keep]
        y_val_model = np.asarray([original_to_model[int(y)] for y in y_val[keep]], dtype=np.int64)
        return x_train, y_train_model, x_val_model, y_val_model

    def _supervised_examples(self, records: Iterable[dict]):
        for record in records:
            family = record.get("family")
            seq = list(record["steps"])
            for i in range(len(seq) + 1):
                target = seq[i] if i < len(seq) else EOS
                if target not in self.step_to_label:
                    continue
                yield family, seq[:i], self.step_to_label[target]

    def _sample_examples(self, examples, max_examples: int, seed_offset: int = 0):
        rng = random.Random(self.config.seed + seed_offset)
        samples = []
        seen = 0
        for example in examples:
            seen += 1
            if max_examples == 0 or len(samples) < max_examples:
                samples.append(example)
                continue
            j = rng.randrange(seen)
            if j < max_examples:
                samples[j] = example
        return samples

    def _build_dataset(
        self,
        records: list[dict],
        max_examples: int = 250_000,
        sample_strategy: str = "uniform",
        family_dropout: float = 0.0,
    ):
        if sample_strategy == "uniform" or max_examples == 0:
            samples = self._sample_examples(self._supervised_examples(records), max_examples)
        elif sample_strategy == "family_balanced":
            grouped: dict[str, list[dict]] = {}
            for record in records:
                grouped.setdefault(record.get("family") or UNK_FAMILY, []).append(record)
            per_family = max(1, math.ceil(max_examples / max(len(grouped), 1)))
            samples = []
            for idx, family in enumerate(sorted(grouped)):
                samples.extend(
                    self._sample_examples(
                        self._supervised_examples(grouped[family]),
                        per_family,
                        seed_offset=idx + 1,
                    )
                )
            if len(samples) > max_examples:
                rng = random.Random(self.config.seed + 10_000)
                rng.shuffle(samples)
                samples = samples[:max_examples]
        else:
            raise ValueError(f"Unknown sample_strategy={sample_strategy!r}")

        if family_dropout > 0:
            rng = random.Random(self.config.seed + 20_000)
            dropout = min(max(float(family_dropout), 0.0), 1.0)
            samples = [
                (None if rng.random() < dropout else family, prefix, target)
                for family, prefix, target in samples
            ]

        x = np.asarray([self.featurize(prefix, family) for family, prefix, _ in samples], dtype=np.float32)
        y = np.asarray([target for _, _, target in samples], dtype=np.int64)
        counts = np.bincount(y, minlength=len(self.label_steps))
        total = counts.sum() or 1
        self.class_priors = {i: float(count / total) for i, count in enumerate(counts)}
        return x, y

    def _step_id(self, step: str) -> int:
        if self.vocab is None:
            return Vocab.UNK
        return self.vocab.stoi.get(step, Vocab.UNK)

    def _family_id(self, family: str | None) -> int:
        return self.family_to_id.get(family or UNK_FAMILY, self.family_to_id[UNK_FAMILY])

    def _count_matching(self, prefix: list[str], predicate) -> int:
        return sum(1 for step in prefix if predicate(step))

    def _seen_matching(self, prefix: list[str], predicate) -> float:
        return 1.0 if any(predicate(step) for step in prefix) else 0.0

    def _last_distance_norm(self, prefix: list[str], predicate) -> float:
        for idx in range(len(prefix) - 1, -1, -1):
            if predicate(prefix[idx]):
                return min(len(prefix) - 1 - idx, self.config.max_seq_len) / self.config.max_seq_len
        return 1.0

    def _op_flags(self, step: str | None) -> list[float]:
        if not step:
            return [0.0 for _ in OP_PREDICATES]
        return [1.0 if predicate(step) else 0.0 for _, predicate in OP_PREDICATES]

    def _extra_features(self, prefix: list[str]) -> list[float]:
        prefix_len = max(len(prefix), 1)
        unique_steps = len(set(prefix))
        litho_levels = [R.align_level(step) for step in prefix]
        litho_levels = [level for level in litho_levels if level is not None]
        last_litho = litho_levels[-1] if litho_levels else 0
        max_litho = max(litho_levels, default=0)

        features = [
            float(unique_steps),
            unique_steps / prefix_len,
            max(len(prefix) - unique_steps, 0) / prefix_len,
            float(last_litho),
            float(max_litho - last_litho),
        ]
        features.extend(float(self._count_matching(prefix, predicate)) for _, predicate in EXTRA_COUNT_PREDICATES)
        features.extend(self._seen_matching(prefix, predicate) for _, predicate in EXTRA_COUNT_PREDICATES)
        features.extend(self._op_flags(prefix[-1] if prefix else None))
        features.extend(self._op_flags(prefix[-2] if len(prefix) >= 2 else None))
        features.extend(self._last_distance_norm(prefix, predicate) for _, predicate in RECENCY_PREDICATES)
        return features

    def featurize(self, prefix: list[str], family: str | None = None) -> list[float]:
        ids = [self._step_id(step) for step in prefix]
        lag_ids = []
        for offset in range(1, self.config.context_size + 1):
            lag_ids.append(ids[-offset] if len(ids) >= offset else Vocab.BOS)
        last2 = tuple(reversed((lag_ids + [Vocab.BOS, Vocab.BOS])[:2]))
        last3 = tuple(reversed((lag_ids + [Vocab.BOS, Vocab.BOS, Vocab.BOS])[:3]))

        categories = {
            "clean": sum(1 for step in prefix if R.is_clean(step)),
            "deposit": sum(1 for step in prefix if R.is_deposit(step)),
            "etch": sum(1 for step in prefix if R.is_etch(step)),
            "implant": sum(1 for step in prefix if R.is_implant(step)),
            "cmp": sum(1 for step in prefix if R.is_cmp(step)),
            "test": sum(1 for step in prefix if R.is_electrical_test(step)),
            "passivation": sum(1 for step in prefix if R.is_passivation(step) or R.is_cure(step)),
            "backside": sum(1 for step in prefix if "BACKSIDE" in step.upper()),
        }
        litho_levels = [R.align_level(step) for step in prefix]
        max_litho = max([lvl for lvl in litho_levels if lvl is not None], default=0)

        features: list[float] = [
            self._family_id(family),
            min(len(prefix), self.config.max_seq_len),
            min(len(prefix), self.config.max_seq_len) / self.config.max_seq_len,
            max_litho,
            categories["clean"],
            categories["deposit"],
            categories["etch"],
            categories["implant"],
            categories["cmp"],
            categories["test"],
            categories["passivation"],
            categories["backside"],
            1.0 if categories["clean"] else 0.0,
            1.0 if categories["deposit"] else 0.0,
            1.0 if categories["etch"] else 0.0,
            1.0 if categories["implant"] else 0.0,
            1.0 if categories["cmp"] else 0.0,
            1.0 if categories["test"] else 0.0,
            1.0 if categories["passivation"] else 0.0,
            1.0 if categories["backside"] else 0.0,
        ]
        features.extend(lag_ids)
        features.append(self.bigram_to_id.get(last2, 0))
        features.append(self.trigram_to_id.get(last3, 0))
        if self._feature_version >= 2:
            features.extend(self._extra_features(prefix))
        return features

    def _proba_rows_to_dicts(self, probs, classes) -> list[dict[str, float]]:
        outputs: list[dict[str, float]] = []
        for row in probs:
            out: dict[str, float] = {}
            for cls, prob in zip(classes, row):
                model_idx = int(cls)
                if 0 <= model_idx < len(self.model_label_ids):
                    label_idx = self.model_label_ids[model_idx]
                    if 0 <= label_idx < len(self.label_steps):
                        out[self.label_steps[label_idx]] = float(prob)
            outputs.append(out)
        return outputs

    def _predict_proba_many(self, prefixes: list[list[str]], family: str | None = None) -> list[dict[str, float]]:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        if not hasattr(self, "_predict_cache"):
            self._predict_cache = {}

        outputs: list[dict[str, float] | None] = [None] * len(prefixes)
        missing = []
        missing_keys = []
        for idx, prefix in enumerate(prefixes):
            key = (family, tuple(prefix))
            cached = self._predict_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
            else:
                missing.append((idx, prefix))
                missing_keys.append(key)

        if missing:
            x = np.asarray([self.featurize(prefix, family) for _, prefix in missing], dtype=np.float32)
            probs = self.estimator.predict_proba(x)
            classes = getattr(self.estimator, "classes_", np.arange(probs.shape[1]))
            for (idx, _), key, out in zip(missing, missing_keys, self._proba_rows_to_dicts(probs, classes)):
                self._predict_cache[key] = out
                outputs[idx] = out

        return [out or {} for out in outputs]

    def _predict_proba(self, prefix: list[str], family: str | None = None) -> dict[str, float]:
        return self._predict_proba_many([prefix], family=family)[0]

    def _predict_label_matrix(self, prefixes: list[list[str]], family: str | None = None):
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray([self.featurize(prefix, family) for prefix in prefixes], dtype=np.float32)
        probs = self.estimator.predict_proba(x)
        classes = list(getattr(self.estimator, "classes_", range(probs.shape[1])))
        model_class_to_pos = {int(cls): pos for pos, cls in enumerate(classes)}
        original_to_model = {label_id: model_idx for model_idx, label_id in enumerate(self.model_label_ids)}
        return probs, model_class_to_pos, original_to_model

    def _prob_for_targets(self, prefixes: list[list[str]], targets: list[str], family: str | None = None) -> list[float]:
        probs, model_class_to_pos, original_to_model = self._predict_label_matrix(prefixes, family=family)
        out = []
        for row, target in enumerate(targets):
            original_label = self.step_to_label.get(target)
            model_label = original_to_model.get(original_label)
            pos = model_class_to_pos.get(model_label)
            out.append(float(probs[row, pos]) if pos is not None else 1e-9)
        return out

    def top_k_next(self, context: list[str], k: int = 5, family: str | None = None) -> list[tuple[str, float]]:
        probs = self._predict_proba(context, family)
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        out = [(step, prob) for step, prob in ranked if step != EOS]
        return out[:k]

    def _completion_score(self, logp: float, generated_len: int) -> float:
        length_norm = getattr(self.config, "beam_length_norm", 0.7)
        denom = max(generated_len, 1) ** length_norm
        return logp / denom

    def _rank_completion_candidates(self, candidates):
        rule_penalty = getattr(self.config, "beam_rule_penalty", 4.0)

        def score(item):
            seq, out, logp, done = item
            penalty = 0.0
            if out and not done and R.attribute_anomaly(seq) is not None:
                penalty += rule_penalty
            if done and out and out[-1] == "SHIP LOT":
                penalty -= 0.75
            return self._completion_score(logp - penalty, len(out))

        return sorted(candidates, key=score, reverse=True)

    def complete_sequence_beam(
        self,
        partial: list[str],
        max_steps: int = 200,
        family: str | None = None,
        beam_width: int | None = None,
        branching: int | None = None,
    ) -> list[str]:
        """Beam-search completion over next-step probabilities.

        Greedy decoding commits to one early choice. Beam search keeps several
        plausible continuations alive, then returns the highest-scoring full
        candidate by normalized log probability plus light process-rule bias.
        """
        beam_width = beam_width or getattr(self.config, "beam_width", 5)
        branching = branching or getattr(self.config, "beam_branching", 8)
        if beam_width <= 1:
            return self.complete_sequence_greedy(partial, max_steps=max_steps, family=family)

        beams = [(list(partial), [], 0.0, False)]  # seq, generated, logp, done
        completed = []

        for _ in range(max_steps):
            expanded = []
            for seq, out, logp, done in beams:
                if done:
                    expanded.append((seq, out, logp, done))
                    completed.append((seq, out, logp, done))
                    continue

                probs = self._predict_proba(seq, family)
                ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:branching]
                if not ranked:
                    expanded.append((seq, out, logp, True))
                    continue

                for step, prob in ranked:
                    p = max(float(prob), 1e-9)
                    next_logp = logp + math.log(p)
                    if step == EOS:
                        expanded.append((seq, out, next_logp, True))
                        completed.append((seq, out, next_logp, True))
                        continue
                    next_seq = seq + [step]
                    next_out = out + [step]
                    done_next = step == "SHIP LOT"
                    expanded.append((next_seq, next_out, next_logp, done_next))
                    if done_next:
                        completed.append((next_seq, next_out, next_logp, done_next))

            beams = self._rank_completion_candidates(expanded)[:beam_width]
            if all(done for _, _, _, done in beams):
                break

        candidates = completed or beams
        return self._rank_completion_candidates(candidates)[0][1] if candidates else []

    def complete_sequence(
        self,
        partial: list[str],
        max_steps: int = 200,
        family: str | None = None,
    ) -> list[str]:
        return self.complete_sequence_beam(partial, max_steps=max_steps, family=family)

    def complete_sequence_greedy(
        self,
        partial: list[str],
        max_steps: int = 200,
        family: str | None = None,
    ) -> list[str]:
        seq = list(partial)
        out = []
        for _ in range(max_steps):
            ranked = self.top_k_next(seq, k=1, family=family)
            if not ranked:
                break
            step = ranked[0][0]
            if step == EOS:
                break
            out.append(step)
            seq.append(step)
            if step == "SHIP LOT":
                break
        return out

    def sequence_log_prob(self, sequence: list[str], family: str | None = None) -> float:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        seq = list(sequence)
        prefixes = []
        targets = []
        for i in range(len(seq) + 1):
            prefixes.append(seq[:i])
            targets.append(seq[i] if i < len(seq) else EOS)

        logp = sum(math.log(max(p, 1e-9)) for p in self._prob_for_targets(prefixes, targets, family=family))
        return logp / max(len(targets), 1)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._predict_cache = {}
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self, f)

    def _expected_feature_count(self) -> int | None:
        if self.estimator is None:
            return None
        if hasattr(self.estimator, "n_features_in_"):
            try:
                return int(self.estimator.n_features_in_)
            except (TypeError, ValueError):
                pass
        if hasattr(self.estimator, "get_booster"):
            try:
                return int(self.estimator.get_booster().num_features())
            except Exception:
                pass
        return None

    def _align_feature_version_to_estimator(self):
        expected = self._expected_feature_count()
        if not expected:
            return
        if len(self.featurize([], None)) == expected:
            return
        original = getattr(self.config, "feature_version", None)
        for version in (1, 2):
            setattr(self.config, "feature_version", version)
            if len(self.featurize([], None)) == expected:
                return
        if original is not None:
            setattr(self.config, "feature_version", original)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        if hasattr(model, "_align_feature_version_to_estimator"):
            model._align_feature_version_to_estimator()
        model._predict_cache = {}
        return model


class XGBoostStepModel(BoostingStepModel):
    model_type = "xgboost"
    NATIVE_MODEL_FILENAME = "xgboost_model.json"
    METADATA_FILENAME = "boosting_metadata.json"
    CLASS_MAP_FILENAME = "class_map.json"

    def _fit_estimator(self, x_train, y_train, x_val, y_val, iterations, device, verbose, **params):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost>=2.0.0 to train/use XGBoostStepModel") from exc

        xgb_params = {
            "objective": "multi:softprob",
            "num_class": len(self.model_label_ids),
            "n_estimators": iterations,
            "tree_method": "hist",
            "max_depth": 8,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 50,
            "random_state": self.config.seed,
            "n_jobs": -1,
        }
        if device == "cuda":
            xgb_params["device"] = "cuda"
        xgb_params.update(params)
        if x_val is None or y_val is None or not len(y_val):
            xgb_params.pop("early_stopping_rounds", None)
        self.estimator = XGBClassifier(**xgb_params)
        fit_kwargs = {"verbose": verbose}
        if x_val is not None and y_val is not None and len(y_val):
            fit_kwargs["eval_set"] = [(x_val, y_val)]
        self.estimator.fit(x_train, y_train, **fit_kwargs)

    def _class_map(self) -> dict:
        classes = []
        for estimator_class_id, label_id in enumerate(self.model_label_ids):
            classes.append(
                {
                    "estimator_class_id": estimator_class_id,
                    "label_id": label_id,
                    "step": self.label_steps[label_id],
                }
            )
        return {
            "format_version": 1,
            "classes": classes,
            "label_steps": self.label_steps,
            "step_to_label": self.step_to_label,
        }

    def _native_metadata(self) -> dict:
        if self.vocab is None:
            raise RuntimeError("model vocabulary is not initialized")
        return {
            "format_version": 1,
            "model_type": self.model_type,
            "config": asdict(self.config),
            "vocab_itos": self.vocab.itos,
            "family_to_id": self.family_to_id,
            "bigram_to_id": [[*key, value] for key, value in self.bigram_to_id.items()],
            "trigram_to_id": [[*key, value] for key, value in self.trigram_to_id.items()],
            "label_steps": self.label_steps,
            "model_label_ids": self.model_label_ids,
            "class_priors": [[key, value] for key, value in self.class_priors.items()],
            "feature_names": self.feature_names,
        }

    def save_native(self, path: str):
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        os.makedirs(path, exist_ok=True)
        self.estimator.save_model(os.path.join(path, self.NATIVE_MODEL_FILENAME))
        with open(os.path.join(path, self.CLASS_MAP_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self._class_map(), f, indent=2)
        with open(os.path.join(path, self.METADATA_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self._native_metadata(), f, indent=2)

    def save(self, path: str):
        super().save(path)
        self.save_native(path)

    @classmethod
    def _load_native(cls, path: str):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost>=2.0.0 to load XGBoostStepModel") from exc

        with open(os.path.join(path, cls.METADATA_FILENAME), encoding="utf-8") as f:
            metadata = json.load(f)
        config_keys = {field.name for field in fields(BoostingConfig)}
        config = BoostingConfig(**{key: value for key, value in metadata["config"].items() if key in config_keys})
        model = cls(config)
        model.vocab = Vocab(steps=[])
        model.vocab.itos = metadata["vocab_itos"]
        model.vocab.stoi = {step: idx for idx, step in enumerate(model.vocab.itos)}
        model.family_to_id = {key: int(value) for key, value in metadata["family_to_id"].items()}
        model.bigram_to_id = {tuple(row[:-1]): int(row[-1]) for row in metadata["bigram_to_id"]}
        model.trigram_to_id = {tuple(row[:-1]): int(row[-1]) for row in metadata["trigram_to_id"]}
        model.label_steps = metadata["label_steps"]
        model.step_to_label = {step: idx for idx, step in enumerate(model.label_steps)}
        model.model_label_ids = [int(value) for value in metadata["model_label_ids"]]
        model.class_priors = {int(key): float(value) for key, value in metadata.get("class_priors", [])}
        model.estimator = XGBClassifier()
        model.estimator.load_model(os.path.join(path, cls.NATIVE_MODEL_FILENAME))
        model._align_feature_version_to_estimator()
        model._predict_cache = {}
        return model

    @classmethod
    def load(cls, path: str):
        pickle_path = os.path.join(path, "model.pkl")
        native_path = os.path.join(path, cls.NATIVE_MODEL_FILENAME)
        metadata_path = os.path.join(path, cls.METADATA_FILENAME)
        if os.path.isfile(pickle_path):
            return super().load(path)
        if os.path.isfile(native_path) and os.path.isfile(metadata_path):
            return cls._load_native(path)
        raise FileNotFoundError(f"No XGBoost checkpoint found under {path}")


class CatBoostStepModel(BoostingStepModel):
    model_type = "catboost"

    def _catboost_frame(self, x):
        import pandas as pd

        df = pd.DataFrame(x, columns=self.feature_names)
        for idx in self.cat_feature_indices:
            col = self.feature_names[idx]
            df[col] = df[col].round().astype("int64").astype("category")
        return df

    def _fit_estimator(self, x_train, y_train, x_val, y_val, iterations, device, verbose, **params):
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install catboost>=1.2.0 to train/use CatBoostStepModel") from exc

        cat_params = {
            "loss_function": "MultiClass",
            "iterations": iterations,
            "depth": 8,
            "learning_rate": 0.08,
            "random_seed": self.config.seed,
            "auto_class_weights": "Balanced",
            "verbose": verbose,
        }
        if device == "cuda":
            cat_params["task_type"] = "GPU"
        cat_params.update(params)
        self.estimator = CatBoostClassifier(**cat_params)

        x_train_cb = self._catboost_frame(x_train)
        eval_set = None
        if x_val is not None and y_val is not None and len(y_val):
            eval_set = (self._catboost_frame(x_val), y_val)

        self.estimator.fit(
            x_train_cb,
            y_train,
            cat_features=[self.feature_names[idx] for idx in self.cat_feature_indices],
            eval_set=eval_set,
            use_best_model=eval_set is not None,
        )

    def _predict_proba_many(self, prefixes: list[list[str]], family: str | None = None) -> list[dict[str, float]]:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        if not hasattr(self, "_predict_cache"):
            self._predict_cache = {}

        outputs: list[dict[str, float] | None] = [None] * len(prefixes)
        missing = []
        missing_keys = []
        for idx, prefix in enumerate(prefixes):
            key = (family, tuple(prefix))
            cached = self._predict_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
            else:
                missing.append((idx, prefix))
                missing_keys.append(key)

        if missing:
            x = np.asarray([self.featurize(prefix, family) for _, prefix in missing], dtype=np.float32)
            probs = self.estimator.predict_proba(self._catboost_frame(x))
            classes = getattr(self.estimator, "classes_", np.arange(probs.shape[1]))
            for (idx, _), key, out in zip(missing, missing_keys, self._proba_rows_to_dicts(probs, classes)):
                self._predict_cache[key] = out
                outputs[idx] = out

        return [out or {} for out in outputs]

    def _predict_proba(self, prefix: list[str], family: str | None = None) -> dict[str, float]:
        return self._predict_proba_many([prefix], family=family)[0]

    def _predict_label_matrix(self, prefixes: list[list[str]], family: str | None = None):
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray([self.featurize(prefix, family) for prefix in prefixes], dtype=np.float32)
        probs = self.estimator.predict_proba(self._catboost_frame(x))
        classes = list(getattr(self.estimator, "classes_", range(probs.shape[1])))
        model_class_to_pos = {int(cls): pos for pos, cls in enumerate(classes)}
        original_to_model = {label_id: model_idx for model_idx, label_id in enumerate(self.model_label_ids)}
        return probs, model_class_to_pos, original_to_model

    def sequence_log_prob(self, sequence: list[str], family: str | None = None) -> float:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        seq = list(sequence)
        prefixes = [seq[:i] for i in range(len(seq) + 1)]
        targets = [seq[i] if i < len(seq) else EOS for i in range(len(seq) + 1)]

        logp = sum(math.log(max(p, 1e-9)) for p in self._prob_for_targets(prefixes, targets, family=family))
        return logp / max(len(targets), 1)


class BoostingEnsembleModel(BoostingStepModel):
    """Probability-average ensemble over saved boosting checkpoints."""

    model_type = "boosting_ensemble"

    def __init__(self, models: list[BoostingStepModel], weights: list[float] | None = None):
        if not models:
            raise ValueError("BoostingEnsembleModel requires at least one member model")
        super().__init__(config=models[0].config)
        self.models = models
        raw_weights = weights or [1.0 for _ in models]
        if len(raw_weights) != len(models):
            raise ValueError("ensemble weights must match number of models")
        total = sum(raw_weights) or 1.0
        self.weights = [float(weight) / total for weight in raw_weights]

    def _predict_proba_many(self, prefixes: list[list[str]], family: str | None = None) -> list[dict[str, float]]:
        if not hasattr(self, "_predict_cache"):
            self._predict_cache = {}
        outputs: list[dict[str, float] | None] = [None] * len(prefixes)
        missing = []
        missing_keys = []
        for idx, prefix in enumerate(prefixes):
            key = (family, tuple(prefix))
            cached = self._predict_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
            else:
                missing.append((idx, prefix))
                missing_keys.append(key)

        if missing:
            combined = [dict() for _ in missing]
            missing_prefixes = [prefix for _, prefix in missing]
            for model, weight in zip(self.models, self.weights):
                for row, probs in enumerate(model._predict_proba_many(missing_prefixes, family=family)):
                    out = combined[row]
                    for step, prob in probs.items():
                        out[step] = out.get(step, 0.0) + weight * float(prob)

            for (idx, _), key, out in zip(missing, missing_keys, combined):
                total = sum(out.values()) or 1.0
                normalized = {step: prob / total for step, prob in out.items()}
                self._predict_cache[key] = normalized
                outputs[idx] = normalized

        return [out or {} for out in outputs]

    def _predict_proba(self, prefix: list[str], family: str | None = None) -> dict[str, float]:
        return self._predict_proba_many([prefix], family=family)[0]

    def sequence_log_prob(self, sequence: list[str], family: str | None = None) -> float:
        seq = list(sequence)
        prefixes = [seq[:i] for i in range(len(seq) + 1)]
        targets = [seq[i] if i < len(seq) else EOS for i in range(len(seq) + 1)]
        probs_by_prefix = self._predict_proba_many(prefixes, family=family)
        logp = 0.0
        for probs, target in zip(probs_by_prefix, targets):
            logp += math.log(max(float(probs.get(target, 1e-9)), 1e-9))
        return logp / max(len(targets), 1)

    @classmethod
    def load_many(cls, paths: list[str], weights: list[float] | None = None):
        models = [XGBoostStepModel.load(path) for path in paths]
        return cls(models, weights=weights)


class BoostingTaskHybridModel:
    """Route each task to the strongest boosting member for that task."""

    model_type = "boosting_task_hybrid"

    def __init__(
        self,
        next_model: BoostingStepModel,
        completion_model: BoostingStepModel,
        anomaly_model: BoostingStepModel,
    ):
        self.next_model = next_model
        self.completion_model = completion_model
        self.anomaly_model = anomaly_model
        self.config = getattr(completion_model, "config", getattr(next_model, "config", BoostingConfig()))

    def top_k_next(self, context: list[str], k: int = 5, family: str | None = None) -> list[tuple[str, float]]:
        return self.next_model.top_k_next(context, k=k, family=family)

    def complete_sequence(self, partial: list[str], max_steps: int = 200, family: str | None = None) -> list[str]:
        return self.completion_model.complete_sequence(partial, max_steps=max_steps, family=family)

    def sequence_log_prob(self, sequence: list[str], family: str | None = None) -> float:
        return self.anomaly_model.sequence_log_prob(sequence, family=family)

    @staticmethod
    def _load_single(path: str):
        return XGBoostStepModel.load(path)

    @classmethod
    def load(
        cls,
        next_paths: list[str],
        completion_path: str,
        anomaly_path: str,
        next_weights: list[float] | None = None,
    ):
        next_model = (
            BoostingEnsembleModel.load_many(next_paths, weights=next_weights)
            if len(next_paths) > 1
            else cls._load_single(next_paths[0])
        )
        return cls(
            next_model=next_model,
            completion_model=cls._load_single(completion_path),
            anomaly_model=cls._load_single(anomaly_path),
        )
