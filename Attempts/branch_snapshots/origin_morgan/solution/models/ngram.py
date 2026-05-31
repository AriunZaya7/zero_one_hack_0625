"""
ngram.py
========
N-gram baseline over process-step strings with back-off (n -> n-1 -> ... -> unigram).
Steps are kept as strings (readable). Default n=3 (trigram).

Serves all three tasks:
  - Task 1: top_k_next(context, k)
  - Task 2: complete_sequence(partial)
  - Task 3: sequence_log_prob(sequence)   (lower = more anomalous)

Smoke test:
    python -m solution.models.ngram --data_dir training_data
"""

import json
import math
import pickle
import argparse
from collections import defaultdict, Counter

BOS = "<BOS>"
EOS = "<EOS>"


class NGramModel:
    def __init__(self, n: int = 3, smoothing: float = 1e-6):
        self.n = n
        self.smoothing = smoothing
        # counts[order][context_tuple][next_step] -> count, for orders 1..n
        self.counts = {order: defaultdict(Counter) for order in range(1, n + 1)}
        self.vocab: set[str] = set()

    def fit(self, sequences: list[list[str]]):
        for seq in sequences:
            padded = [BOS] * (self.n - 1) + list(seq) + [EOS]
            for i in range(len(padded)):
                nxt = padded[i]
                if nxt != BOS:
                    self.vocab.add(nxt)
                for order in range(1, self.n + 1):
                    if i - (order - 1) < 0:
                        continue
                    context = tuple(padded[i - (order - 1):i])
                    self.counts[order][context][nxt] += 1
        self.vocab.discard(EOS)
        return self

    def _backoff_counter(self, context: list[str]) -> Counter:
        for order in range(self.n, 0, -1):
            ctx = tuple(([BOS] * (self.n - 1) + list(context))[-(order - 1):]) if order > 1 else ()
            counter = self.counts[order].get(ctx)
            if counter:
                return counter
        # final fallback: unigram totals
        total = Counter()
        for c in self.counts[1].values():
            total.update(c)
        return total

    def next_step_probs(self, context: list[str]) -> dict[str, float]:
        counter = self._backoff_counter(context)
        total = sum(counter.values())
        vocab_size = len(self.vocab) + 1
        return {
            step: (counter.get(step, 0) + self.smoothing) / (total + self.smoothing * vocab_size)
            for step in counter
        }

    def top_k_next(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        counter = self._backoff_counter(context)
        total = sum(counter.values()) or 1
        ranked = [(s, c / total) for s, c in counter.most_common() if s != EOS]
        return ranked[:k]

    def complete_sequence(self, partial: list[str], max_steps: int = 200) -> list[str]:
        seq = list(partial)
        out: list[str] = []
        for _ in range(max_steps):
            top = self.top_k_next(seq, k=1)
            if not top:
                break
            nxt = top[0][0]
            if nxt in (EOS, "<UNK>"):
                break
            out.append(nxt)
            seq.append(nxt)
            if nxt == "SHIP LOT":
                break
        return out

    def sequence_log_prob(self, sequence: list[str]) -> float:
        """Mean per-step log-prob (length-normalized). Higher = more typical."""
        padded = [BOS] * (self.n - 1) + list(sequence) + [EOS]
        vocab_size = len(self.vocab) + 1
        lp = 0.0
        n_pred = 0
        for i in range(self.n - 1, len(padded)):
            context = padded[i - (self.n - 1):i]
            counter = self._backoff_counter(context)
            total = sum(counter.values())
            count = counter.get(padded[i], 0)
            prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            lp += math.log(prob)
            n_pred += 1
        return lp / max(n_pred, 1)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"n": self.n, "smoothing": self.smoothing,
                         "counts": self.counts, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "NGramModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(n=d["n"], smoothing=d["smoothing"])
        m.counts = d["counts"]
        m.vocab = d["vocab"]
        return m


if __name__ == "__main__":
    from solution.data.loader import load_all_families, train_val_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    fams = load_all_families(args.data_dir)
    all_seqs = [s for seqs in fams.values() for s in seqs]
    tr, va = train_val_split(all_seqs, seed=42)
    m = NGramModel(n=args.n).fit(tr)
    ctx = va[0][:5]
    print("context:", ctx[-3:])
    print("top-5 next:", [s for s, _ in m.top_k_next(ctx, 5)])
    print("seq logprob (val[0]):", round(m.sequence_log_prob(va[0]), 4))
    print("ngram OK")
