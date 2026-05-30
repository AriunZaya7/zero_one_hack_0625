"""
adapters.py
===========
Wrap each model type behind one interface so the task functions and run_eval.py
are model-agnostic:

    top_k_next(context: list[str], k, family=None) -> list[str]
    complete(partial: list[str], family=None) -> list[str]
    seq_log_prob(seq: list[str], family=None) -> float

NGramModel and LLMModel already expose most of this. The transformer needs a
Vocab to map between step strings and token ids, handled by TransformerAdapter.
"""

from __future__ import annotations
from solution.data.vocab import Vocab


class NGramAdapter:
    def __init__(self, model):
        self.m = model

    def top_k_next(self, context, k=5, family=None):
        return [s for s, _ in self.m.top_k_next(context, k)]

    def complete(self, partial, family=None, max_steps=200):
        return self.m.complete_sequence(partial, max_steps=max_steps)

    def seq_log_prob(self, seq, family=None):
        return self.m.sequence_log_prob(seq)


class LLMAdapter:
    def __init__(self, model):
        self.m = model

    def top_k_next(self, context, k=5, family=None):
        return [s for s, _ in self.m.top_k_next(context, k)]

    def complete(self, partial, family=None, max_steps=200):
        try:
            return self.m.complete_sequence(partial, max_steps=max_steps)
        except TypeError:
            return self.m.complete_sequence(partial)

    def seq_log_prob(self, seq, family=None):
        return self.m.sequence_log_prob(seq)


class BoostingAdapter:
    def __init__(self, model):
        self.m = model

    def top_k_next(self, context, k=5, family=None):
        return [s for s, _ in self.m.top_k_next(context, k, family=family)]

    def complete(self, partial, family=None, max_steps=200):
        return self.m.complete_sequence(partial, max_steps=max_steps, family=family)

    def seq_log_prob(self, seq, family=None):
        return self.m.sequence_log_prob(seq, family=family)


class TransformerAdapter:
    def __init__(self, model, vocab: Vocab, device: str = "cpu"):
        import torch
        self.torch = torch
        self.m = model.to(device).eval()
        self.v = vocab
        self.device = device

    def _ids(self, steps):
        return self.torch.tensor([self.v.encode(steps, add_special=True)[:-1]],  # drop EOS for a prefix
                                 device=self.device)

    def top_k_next(self, context, k=5, family=None):
        logits = self.m.next_token_logits(self._ids(context))[0]
        topk = self.torch.topk(logits, min(k + 4, logits.shape[-1])).indices.tolist()
        out = []
        for i in topk:
            if i < self.v.OFFSET:
                continue
            out.append(self.v.itos[i])
            if len(out) == k:
                break
        return out

    def complete(self, partial, family=None, max_steps=200):
        prompt = self._ids(partial)
        gen = self.m.generate(prompt, max_new_tokens=max_steps, eos_id=self.v.EOS, greedy=True)
        new_ids = gen[0, prompt.shape[1]:].tolist()
        return self.v.decode(new_ids, strip_special=True)

    def seq_log_prob(self, seq, family=None):
        ids = self.torch.tensor([self.v.encode(seq, add_special=True)], device=self.device)
        return float(self.m.sequence_log_prob(ids)[0].item())


def make_adapter(kind: str, model, vocab=None, device="cpu"):
    if kind == "ngram":
        return NGramAdapter(model)
    if kind == "llm":
        return LLMAdapter(model)
    if kind in {"xgboost", "catboost", "boosting"}:
        return BoostingAdapter(model)
    if kind == "transformer":
        return TransformerAdapter(model, vocab, device)
    raise ValueError(f"unknown model kind: {kind}")
