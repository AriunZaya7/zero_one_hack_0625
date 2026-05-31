"""Count-based n-gram baseline with stupid backoff.

This is the NON-NEURAL reference every task is compared against. It is fast,
dependency-free, and surprisingly strong on highly structured sequences --
which makes it the honest bar the transformer must clear.

It serves all three tasks through one shared interface (mirrors the GPT
wrapper so eval code is model-agnostic):
  next_step_ranking(prefix, k) -> top-k next steps
  complete(prefix)             -> greedy roll-out to <eos>
  sequence_surprisal(seq)      -> mean negative log-prob (anomaly score)

Default: trigram (n=3) with stupid-backoff discount alpha=0.4.
Tune n in {2,3,4} and alpha in [0.3,0.7] on a validation split.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class NGramModel:
    def __init__(self, n: int = 3, alpha: float = 0.4):
        self.n = n
        self.alpha = alpha
        # counts[order][context_tuple][token] -> count ; order 0 == unigram
        self.counts: list[dict] = [defaultdict(Counter) for _ in range(n)]
        self.vocab: set[str] = set()
        self.total: int = 0

    def fit(self, sequences):
        for seq in sequences:
            s = ["<bos>"] + list(seq) + ["<eos>"]
            self.vocab.update(s)
            for i, tok in enumerate(s):
                self.total += 1
                for order in range(self.n):
                    if i - order < 0:
                        break
                    ctx = tuple(s[i - order:i])
                    self.counts[order][ctx][tok] += 1
        return self

    def _prob(self, ctx: tuple, tok: str) -> float:
        max_order = min(self.n - 1, len(ctx))
        for order in range(max_order, -1, -1):
            c = tuple(ctx[len(ctx) - order:]) if order > 0 else ()
            table = self.counts[order].get(c)
            if table and tok in table:
                denom = sum(table.values())
                weight = self.alpha ** (max_order - order)
                return weight * table[tok] / denom
        # unigram floor (additive smoothing)
        return 1.0 / (self.total + len(self.vocab) + 1)

    def next_step_ranking(self, prefix, k: int = 5):
        ctx = tuple(["<bos>"] + list(prefix))
        scored = [(t, self._prob(ctx, t)) for t in self.vocab if t != "<bos>"]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:k]]

    def complete(self, prefix, max_len: int = 200):
        seq = list(prefix)
        for _ in range(max_len):
            nxt = self.next_step_ranking(seq, k=1)[0]
            if nxt == "<eos>":
                break
            seq.append(nxt)
        return seq

    def sequence_surprisal(self, seq) -> float:
        s = ["<bos>"] + list(seq) + ["<eos>"]
        total = 0.0
        for i in range(1, len(s)):
            p = max(self._prob(tuple(s[:i]), s[i]), 1e-12)
            total += -math.log(p)
        return total / (len(s) - 1)
