"""
Classical boosted-tree next-step models.

Both XGBoost and CatBoost are trained as multiclass classifiers over
step-level next-token prediction. They expose the same methods as the other
solution models: top_k_next(), complete_sequence(), and sequence_log_prob().
"""

from __future__ import annotations

import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from solution.data.vocab import Vocab
from solution.eval import rules as R

BOS = "<BOS>"
EOS = "<EOS>"
UNK_FAMILY = "<UNK_FAMILY>"


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
        return names

    def fit(
        self,
        train_records: list[dict],
        val_records: list[dict] | None = None,
        max_train_examples: int = 250_000,
        max_val_examples: int = 50_000,
        iterations: int = 1200,
        device: str = "cpu",
        verbose: bool | int = True,
        **params,
    ):
        self._build_metadata(train_records)
        x_train, y_train = self._build_dataset(train_records, max_examples=max_train_examples)
        x_val = y_val = None
        if val_records:
            x_val, y_val = self._build_dataset(val_records, max_examples=max_val_examples)
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

    def _build_dataset(self, records: list[dict], max_examples: int = 250_000):
        rng = random.Random(self.config.seed)
        samples = []
        seen = 0
        for example in self._supervised_examples(records):
            seen += 1
            if max_examples == 0 or len(samples) < max_examples:
                samples.append(example)
                continue
            j = rng.randrange(seen)
            if j < max_examples:
                samples[j] = example

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
        return features

    def _predict_proba(self, prefix: list[str], family: str | None = None) -> dict[str, float]:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray([self.featurize(prefix, family)], dtype=np.float32)
        probs = self.estimator.predict_proba(x)[0]
        classes = getattr(self.estimator, "classes_", np.arange(len(probs)))
        out: dict[str, float] = {}
        for cls, prob in zip(classes, probs):
            model_idx = int(cls)
            if 0 <= model_idx < len(self.model_label_ids):
                label_idx = self.model_label_ids[model_idx]
                if 0 <= label_idx < len(self.label_steps):
                    out[self.label_steps[label_idx]] = float(prob)
        return out

    def top_k_next(self, context: list[str], k: int = 5, family: str | None = None) -> list[tuple[str, float]]:
        probs = self._predict_proba(context, family)
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        out = [(step, prob) for step, prob in ranked if step != EOS]
        return out[:k]

    def complete_sequence(
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

        x = np.asarray([self.featurize(prefix, family) for prefix in prefixes], dtype=np.float32)
        probs = self.estimator.predict_proba(x)
        classes = list(getattr(self.estimator, "classes_", range(probs.shape[1])))
        model_class_to_pos = {int(cls): pos for pos, cls in enumerate(classes)}
        original_to_model = {label_id: model_idx for model_idx, label_id in enumerate(self.model_label_ids)}

        logp = 0.0
        for row, target in enumerate(targets):
            original_label = self.step_to_label.get(target)
            model_label = original_to_model.get(original_label)
            pos = model_class_to_pos.get(model_label)
            p = float(probs[row, pos]) if pos is not None else 1e-9
            logp += math.log(max(p, 1e-9))
        return logp / max(len(targets), 1)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            return pickle.load(f)


class XGBoostStepModel(BoostingStepModel):
    model_type = "xgboost"

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
        if x_val is None or y_val is None or not len(y_val):
            xgb_params.pop("early_stopping_rounds", None)
        xgb_params.update(params)
        self.estimator = XGBClassifier(**xgb_params)
        fit_kwargs = {"verbose": verbose}
        if x_val is not None and y_val is not None and len(y_val):
            fit_kwargs["eval_set"] = [(x_val, y_val)]
        self.estimator.fit(x_train, y_train, **fit_kwargs)


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

    def _predict_proba(self, prefix: list[str], family: str | None = None) -> dict[str, float]:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray([self.featurize(prefix, family)], dtype=np.float32)
        probs = self.estimator.predict_proba(self._catboost_frame(x))[0]
        classes = getattr(self.estimator, "classes_", np.arange(len(probs)))
        out: dict[str, float] = {}
        for cls, prob in zip(classes, probs):
            model_idx = int(cls)
            if 0 <= model_idx < len(self.model_label_ids):
                label_idx = self.model_label_ids[model_idx]
                if 0 <= label_idx < len(self.label_steps):
                    out[self.label_steps[label_idx]] = float(prob)
        return out

    def sequence_log_prob(self, sequence: list[str], family: str | None = None) -> float:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        seq = list(sequence)
        prefixes = [seq[:i] for i in range(len(seq) + 1)]
        targets = [seq[i] if i < len(seq) else EOS for i in range(len(seq) + 1)]

        x = np.asarray([self.featurize(prefix, family) for prefix in prefixes], dtype=np.float32)
        probs = self.estimator.predict_proba(self._catboost_frame(x))
        classes = list(getattr(self.estimator, "classes_", range(probs.shape[1])))
        model_class_to_pos = {int(cls): pos for pos, cls in enumerate(classes)}
        original_to_model = {label_id: model_idx for model_idx, label_id in enumerate(self.model_label_ids)}

        logp = 0.0
        for row, target in enumerate(targets):
            original_label = self.step_to_label.get(target)
            model_label = original_to_model.get(original_label)
            pos = model_class_to_pos.get(model_label)
            p = float(probs[row, pos]) if pos is not None else 1e-9
            logp += math.log(max(p, 1e-9))
        return logp / max(len(targets), 1)
