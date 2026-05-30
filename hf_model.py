"""Pretrained HF causal LM (e.g. Qwen2.5) fine-tuned on process sequences.

This is the *transfer-learning* tier (PLAN.md model strategy): a real pretrained
LLM with its NATIVE subword tokenizer, fine-tuned on the step sequences. It is
deliberately NOT the from-scratch GPT -- we keep that as the clean "learned the
grammar from nothing" anchor and use this to measure how far pretraining pushes
the ceiling, especially on the held-out family.

Key trick so it is measured IDENTICALLY to the n-gram and the from-scratch GPT:
the process vocabulary is CLOSED and shared (~198 steps). So next-step prediction
is scored as **closed-vocab ranking**: for a prefix, we rank every candidate step
by the model's log-prob of generating that step's subword tokens. That gives clean
top-k / MRR / surprisal through the same shared interface (PLAN.md §3), even though
the model tokenizes at the subword level.

Text representation: a sequence is the steps joined by newlines, each step ending
in "\n". The newline is the natural step boundary the model predicts after.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEP_SEP = "\n"  # boundary between steps in the text representation


def seq_to_text(steps: list[str]) -> str:
    """Sequence -> training/scoring text. Each step terminated by the separator."""
    return "".join(s + STEP_SEP for s in steps)


def load_hf(name: str, dtype=None):
    """Load a pretrained tokenizer+model. Offline-safe (set HF_HUB_OFFLINE=1)."""
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype or torch.float32)
    return model, tok


class HFModel:
    """Wraps a pretrained causal LM behind the shared interface.

    `step_vocab` is the closed list of candidate steps (from StepTokenizer).
    """

    def __init__(self, model, tok, step_vocab: list[str], device: str | None = None,
                 cand_chunk: int = 64):
        self.model = model
        self.tok = tok
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.cand_chunk = cand_chunk  # candidates scored per batched forward
        # pre-tokenize the candidate steps once (each ends with the separator)
        self.step_vocab = list(step_vocab)
        self._cand_ids = [
            tok(s + STEP_SEP, add_special_tokens=False).input_ids for s in self.step_vocab
        ]

    # -- closed-vocab next-step scoring (the core) ---------------------------
    @torch.no_grad()
    def _score_candidates(self, prefix: list[str]) -> torch.Tensor:
        """Return log P(step | prefix) for every step in step_vocab."""
        prefix_ids = self.tok(seq_to_text(prefix), add_special_tokens=True).input_ids
        p = len(prefix_ids)
        scores = torch.full((len(self.step_vocab),), float("-inf"))
        order = range(0, len(self.step_vocab), self.cand_chunk)
        for start in order:
            chunk = list(range(start, min(start + self.cand_chunk, len(self.step_vocab))))
            cand = [self._cand_ids[i] for i in chunk]
            maxlen = p + max(len(c) for c in cand)
            input_ids, attn = [], []
            for c in cand:
                row = prefix_ids + c
                pad = maxlen - len(row)
                input_ids.append(row + [self.tok.pad_token_id] * pad)
                attn.append([1] * len(row) + [0] * pad)
            x = torch.tensor(input_ids, device=self.device)
            a = torch.tensor(attn, device=self.device)
            logp = torch.log_softmax(self.model(input_ids=x, attention_mask=a).logits, dim=-1)
            for j, i in enumerate(chunk):
                c = self._cand_ids[i]
                # position p-1+t predicts candidate token t
                tot = 0.0
                for t, tokid in enumerate(c):
                    tot += logp[j, p - 1 + t, tokid].item()
                scores[i] = tot
        return scores

    def next_step_ranking(self, prefix: list[str], k: int = 5) -> list[str]:
        scores = self._score_candidates(prefix)
        top = torch.topk(scores, min(k, len(self.step_vocab))).indices.tolist()
        return [self.step_vocab[i] for i in top]

    def complete(self, prefix: list[str], max_len: int = 200) -> list[str]:
        seq = list(prefix)
        for _ in range(max_len):
            nxt = self.next_step_ranking(seq, k=1)[0]
            seq.append(nxt)
            if len(seq) - len(prefix) >= max_len:
                break
            # stop heuristic: SHIP LOT is the canonical terminal step
            if nxt.strip().upper().startswith("SHIP LOT"):
                break
        return seq

    # -- surprisal (Task 3) ---------------------------------------------------
    @torch.no_grad()
    def per_step_surprisal(self, seq: list[str]) -> list[float]:
        """NLL (nats) summed over each step's subword tokens, one value per step."""
        per_step = []
        for i, step in enumerate(seq):
            prefix = seq[:i]
            prefix_ids = self.tok(seq_to_text(prefix), add_special_tokens=True).input_ids
            step_ids = self._cand_ids[None] if False else \
                self.tok(step + STEP_SEP, add_special_tokens=False).input_ids
            ids = prefix_ids + step_ids
            x = torch.tensor([ids], device=self.device)
            logp = torch.log_softmax(self.model(x).logits[0], dim=-1)
            p = len(prefix_ids)
            nll = -sum(logp[p - 1 + t, tokid].item() for t, tokid in enumerate(step_ids))
            per_step.append(nll)
        return per_step

    def sequence_surprisal(self, seq: list[str]) -> float:
        s = self.per_step_surprisal(seq)
        return sum(s) / len(s)


if __name__ == "__main__":
    # smoke test (needs the model cached locally / online)
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    from data import load_all
    from tokenizer import StepTokenizer
    tok_step = StepTokenizer.build_from_sequences(load_all().values())
    steps = sorted(s for s in tok_step.step_to_id if not s.startswith("<"))
    model, hftok = load_hf(name)
    m = HFModel(model, hftok, steps)
    print("ranking:", m.next_step_ranking(["RECEIVE WAFER LOT", "LOT IDENTIFICATION"], k=3))
